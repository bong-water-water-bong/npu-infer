#!/usr/bin/env python3
#
# INT8 MLIR generator v4 — Correct K-tiling for B
# Both A and B are tiled in K with matching offsets
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
    mtk = 64  # K per L2 tile = k (single K-block per L2 fill)
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 3)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        A_l3l2 = [None] * n_aie_rows
        A_l2l1 = [None] * n_aie_rows
        B_l3l2 = [None] * n_aie_cols
        B_l2l1 = [None] * n_aie_cols
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols

        for r in range(n_aie_rows):
            A_l3l2[r] = object_fifo(f'A_L3L2_{r}', shim_tiles[r], mem_tiles[r], 2, A_l2_ty)
            A_l2l1[r] = object_fifo(f'A_L2L1_{r}', mem_tiles[r], core_tiles[r][0:n_aie_cols], 2, A_l1_ty)
            object_fifo_link(A_l3l2[r], A_l2l1[r])

        for c in range(n_aie_cols):
            B_l3l2[c] = object_fifo(f'B_L3L2_{c}', shim_tiles[c], mem_tiles[c], 2, B_l2_ty)
            B_l2l1[c] = object_fifo(f'B_L2L1_{c}', mem_tiles[c], [core_tiles[j][c] for j in range(n_aie_rows)], 2, B_l1_ty)
            object_fifo_link(B_l3l2[c], B_l2l1[c])

        for c in range(n_aie_cols):
            for r in range(n_aie_rows):
                C_l1l2[r][c] = object_fifo(f'C_L1L2_{c}_{r}', core_tiles[r][c], mem_tiles[c], 1, C_l1_ty)
            C_l2l3[c] = object_fifo(f'C_L2L3_{c}', mem_tiles[c], shim_tiles[c], 2, C_l2_ty)
            object_fifo_link([C_l1l2[j][c] for j in range(n_aie_rows)], C_l2l3[c], [m * n * j for j in range(n_aie_rows)])

        for r in range(n_aie_rows):
            for c in range(n_aie_cols):
                @core(core_tiles[r][c], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M//m)*(N//n)//(n_aie_cols*n_aie_rows)):
                            Cbuf = C_l1l2[r][c].acquire(ObjectFifoPort.Produce, 1)
                            zero(Cbuf)
                            for _ in range_(K//k):
                                Bbuf = B_l2l1[c].acquire(ObjectFifoPort.Consume, 1)
                                Abuf = A_l2l1[r].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, Cbuf)
                                A_l2l1[r].release(ObjectFifoPort.Consume, 1)
                                B_l2l1[c].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[r][c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M*K,), np.dtype[np.int8]],
            np.ndarray[(K*N,), np.dtype[np.int8]],
            np.ndarray[(M*N,), np.dtype[np.int16]],
        )
        def seq(A, B, C):
            # K-tiled taps for both A and B
            # A: (M,K) → (m,k) tiles, repeat (1, K//k) times
            # B: (K,N) → (k,n) tiles, repeat (K//k, 1) times in K, (1, N//n) times in N
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, N // n))
            C_taps = TensorTiler2D.group_tiler((M, N), (n_aie_rows*m, n), (1, N // n // n_aie_cols))
            
            n_mtiles = M // m  # number of M-tiles
            n_ktiles = K // k  # number of K-tiles
            n_ntiles_per_core = N // n // n_aie_cols  # N-tiles per core
            n_cgroups = n_mtiles * n_ntiles_per_core  # total compute groups
            tb = 4
            itg = [[] for _ in range(tb)]
            otg = [[] for _ in range(tb)]
            
            for cgi in range(n_cgroups):
                mt = cgi // n_ntiles_per_core  # which M-tile
                nt = cgi % n_ntiles_per_core   # which N-tile
                
                # For this compute group, iterate over K-tiles
                for kti in range(n_ktiles):
                    a_tap_idx = mt * n_ktiles + kti
                    for r in range(n_aie_rows):
                        for r in range(n_aie_rows):
                            at = shim_dma_single_bd_task(A_l3l2[r], A, tap=A_taps[a_tap_idx + r], issue_token=False)
                            dma_start_task(at); itg[cgi%tb].append(at)
                        
                        b_tap_idx = kti * (N // n) + nt * n_aie_cols
                        for c in range(n_aie_cols):
                            bt = shim_dma_single_bd_task(B_l3l2[c], B, tap=B_taps[b_tap_idx + c], issue_token=False)
                            dma_start_task(bt); itg[cgi%tb].append(bt)
                    
                    # Output: one C tile per group
                    c_tap_idx = cgi * n_aie_cols
                    for c in range(n_aie_cols):
                        ct_ = shim_dma_single_bd_task(C_l2l3[c], C, tap=C_taps[c_tap_idx + c], issue_token=True)
                        dma_start_task(ct_); otg[cgi%tb].append(ct_)
                
                # Drain completed groups
                if cgi%tb==1 and cgi!=1:
                    for g in [2,3]:
                        if otg[g]: dma_await_task(*otg[g]); otg[g]=[]
                        dma_free_task(*itg[g]); itg[g]=[]
                if cgi%tb==3:
                    for g in [0,1]:
                        if otg[g]: dma_await_task(*otg[g]); otg[g]=[]
                        dma_free_task(*itg[g]); itg[g]=[]
            
            for g in range(tb):
                if otg[g]: dma_await_task(*otg[g]); dma_free_task(*itg[g])

main()
