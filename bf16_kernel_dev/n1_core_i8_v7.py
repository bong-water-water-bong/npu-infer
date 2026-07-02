#!/usr/bin/env python3
#
# INT8 MLIR generator v7 — WORKING FINAL VERSION
# Per-core A fifos + per-core B fifos, all start at same K-offset
# Uses original v2 approach (m=32, K-loop in core, M-tiling at runtime)
# Each core gets its own mem tile for A that reads the same L3 data
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
    mtk = 512  # K-dim in L2 buffer
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2:]
        
        # Per-core A fifos: each core reads A from same L3 offset via its mem tile
        A_s = [None]*n_aie_cols
        A_c = [None]*n_aie_cols
        for c in range(n_aie_cols):
            A_s[c] = object_fifo(f"A_S{c}", st[c], mt[c], 2, A_l2_ty, None,
                [[(m, k), (mtk // k, m * k), (k, 1)]])
            # Share one A fifo per mem tile → one core per mem tile
            # But the L2→L1 fifo has dimensions that give ALL K-slices to this core
            A_c[c] = object_fifo(f"A_C{c}", mt[c], ct[0][c], 2, A_l1_ty,
                [(mtk // k, m * k // 4), (k // k, k), (m // 4, k), (1, 1)],  # flat
                [[(k, k), (m // 4, m * k), (1, 1)]])  # flat L1
            object_fifo_link(A_s[c], A_c[c])
        
        # Per-core B fifos
        B_s = [None]*n_aie_cols
        B_c = [None]*n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", st[c], mt[c], 2, B_l2_ty)
            B_c[c] = object_fifo(f"B_C{c}", mt[c], ct[0][c], 2, B_l1_ty)
            object_fifo_link(B_s[c], B_c[c])
        
        # C output
        C_c = [None]*n_aie_cols
        C_s = [None]*n_aie_cols
        for c in range(n_aie_cols):
            C_c[c] = object_fifo(f"C_c{c}", ct[0][c], mt[c], 1, C_l1_ty)
            C_s[c] = object_fifo(f"C_s{c}", mt[c], st[c], 2, C_l2_ty)
            object_fifo_link(C_c[c], C_s[c])
        
        # Core body
        for c in range(n_aie_cols):
            @core(ct[0][c], stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M//m)*(N//n)//n_aie_cols):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K//k):
                            Abuf = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M*K,), np.dtype[np.int8]],
            np.ndarray[(K*N,), np.dtype[np.int8]],
            np.ndarray[(M*N,), np.dtype[np.int16]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K//mtk))
            B_taps = TensorTiler2D.group_tiler((1, N*K), (1, n*K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (M//m, N//n))
            n_mt = M//m
            n_nt = N//n
            n_cg = n_mt * (n_nt // n_aie_cols)
            tb = 4
            itg = [[] for _ in range(tb)]
            otg = [[] for _ in range(tb)]
            for gi in range(n_cg):
                mi = gi // (n_nt // n_aie_cols)
                ni = gi % (n_nt // n_aie_cols)
                for c in range(n_aie_cols):
                    at = shim_dma_single_bd_task(A_s[c], A, tap=A_taps[mi], issue_token=False)
                    dma_start_task(at); itg[gi%tb].append(at)
                    bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[ni*n_aie_cols+c], issue_token=False)
                    dma_start_task(bt); itg[gi%tb].append(bt)
                ct_ = shim_dma_single_bd_task(C_s[0], C, tap=C_taps[gi], issue_token=True)
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
