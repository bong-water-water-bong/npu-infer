#!/usr/bin/env python3
#
# INT8 MLIR generator v5 — Working design
# - Each core has its own B fifo (from shim via mem tile)
# - All cores process same K-slice per iteration
# - K-tiling handled at runtime level per compute group
# - M=32 per core, N=128 per core, K=64 per tile
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    n_aie_rows = 1
    dtype_in = np.int8
    dtype_out = np.int16
    
    @device(AIEDevice.npu2)
    def device_body():
        # Types
        A_l2_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 3)]
        st = tiles[0]
        mt = tiles[1]
        ct = tiles[2:]  # list of 1 list of 8 tiles = [[t0, t1, ..., t7]]
        
        # Separate A fifo per core, all reading from same L3 tap
        # Separate B fifo per core, reading from core-specific N-block
        A_s = [[None]*n_aie_cols for _ in range(n_aie_rows)]
        A_c = [[None]*n_aie_cols for _ in range(n_aie_rows)]
        B_s = [[None]*n_aie_cols for _ in range(n_aie_rows)]
        B_c = [[None]*n_aie_cols for _ in range(n_aie_rows)]
        for r in range(n_aie_rows):
            for c in range(n_aie_cols):
                # A: from each shim tile (reads same L3 data, different DMA channel)
                A_s[r][c] = object_fifo(f"A_S{r}{c}", st[r], mt[c], 2, A_l2_ty)
                A_c[r][c] = object_fifo(f"A_C{r}{c}", mt[c], ct[r][c], 2, A_l1_ty)
                object_fifo_link(A_s[r][c], A_c[r][c])
                # B: from core-specific shim for N-block
                B_s[r][c] = object_fifo(f"B_S{r}{c}", st[c], mt[c], 2, B_l2_ty)
                B_c[r][c] = object_fifo(f"B_C{r}{c}", mt[c], ct[r][c], 2, B_l1_ty)
                object_fifo_link(B_s[r][c], B_c[r][c])
        
        # C output: core → mem → shim
        C_c = [[None]*n_aie_cols for _ in range(n_aie_rows)]
        C_s = [None]*n_aie_cols
        for c in range(n_aie_cols):
            for r in range(n_aie_rows):
                C_c[r][c] = object_fifo(f"C_C{c}_{r}", ct[r][c], mt[c], 1, C_l1_ty)
            C_s[c] = object_fifo(f"C_S{c}", mt[c], st[c], 2, C_l2_ty)
            object_fifo_link([C_c[j][c] for j in range(n_aie_rows)], C_s[c])
        
        # Core body
        for r in range(n_aie_rows):
            for c in range(n_aie_cols):
                @core(ct[r][c], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M//m)*(N//n)//(n_aie_cols*n_aie_rows)):
                            Cbuf = C_c[r][c].acquire(ObjectFifoPort.Produce, 1)
                            zero(Cbuf)
                            for _ in range_(K//k):
                                Abuf = A_c[r][c].acquire(ObjectFifoPort.Consume, 1)
                                Bbuf = B_c[r][c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, Cbuf)
                                A_c[r][c].release(ObjectFifoPort.Consume, 1)
                                B_c[r][c].release(ObjectFifoPort.Consume, 1)
                            C_c[r][c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M*K,), np.dtype[np.int8]],
            np.ndarray[(K*N,), np.dtype[np.int8]],
            np.ndarray[(M*N,), np.dtype[np.int16]],
        )
        def seq(A, B, C):
            # Each compute group covers M-tile + N-block + all K-tiles
            # Schedule A+B for each K-tile, then C output
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K//k))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (K//k, N//n))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (M//m, N//n))
            
            n_mt = M//m//n_aie_rows
            n_nt_per_core = N//n//n_aie_cols
            n_kt = K//k
            n_groups = n_mt * n_nt_per_core
            tb = 4
            itg = [[] for _ in range(tb)]
            otg = [[] for _ in range(tb)]
            
            for gi in range(n_groups):
                mi = gi // n_nt_per_core
                ni = gi % n_nt_per_core
                
                # For each K-tile: feed A (same for all cores) and B (per core)
                for kti in range(n_kt):
                    a_tap = A_taps[mi * n_kt + kti]
                    for r in range(n_aie_rows):
                        for c in range(n_aie_cols):
                            at = shim_dma_single_bd_task(A_s[r][c], A, tap=a_tap, issue_token=False)
                            dma_start_task(at); itg[gi%tb].append(at)
                            bt = shim_dma_single_bd_task(B_s[r][c], B, tap=B_taps[kti * (N//n) + ni * n_aie_cols + c], issue_token=False)
                            dma_start_task(bt); itg[gi%tb].append(bt)
                
                # One C output per compute group per core
                for c in range(n_aie_cols):
                    ct_ = shim_dma_single_bd_task(C_s[c], C, tap=C_taps[gi * n_aie_cols + c], issue_token=True)
                    dma_start_task(ct_); otg[gi%tb].append(ct_)
                
                if gi%tb==1 and gi!=1:
                    for g in [2,3]:
                        if otg[g]: dma_await_task(*otg[g]); otg[g]=[]
                        dma_free_task(*itg[g]); itg[g]=[]
                if gi%tb==3:
                    for g in [0,1]:
                        if otg[g]: dma_await_task(*otg[g]); otg[g]=[]
                        dma_free_task(*itg[g]); itg[g]=[]
            
            for g in range(tb):
                if otg[g]: dma_await_task(*otg[g]); dma_free_task(*itg[g])

main()
