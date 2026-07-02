#!/usr/bin/env python3
#
# INT8 MLIR generator v8 — CORRECT BROADCAST
# - Each core has its OWN A fifo (reads same L3 data, same K-slice)
# - Each core has its OWN B fifo (core-specific N-block, same K-slice)
# - K-loop inside core accumulates across all K-tiles
# - M-tiling at runtime level
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
    mtk = 512  # K per L2 tile
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2]
        
        # A and B: per-core fifos, all A read same L3 offset, B read per-N-block
        A_s = [None]*n_aie_cols
        A_c = [None]*n_aie_cols
        B_s = [None]*n_aie_cols
        B_c = [None]*n_aie_cols
        C_c = [None]*n_aie_cols
        C_s = [None]*n_aie_cols
        
        for c in range(n_aie_cols):
            # A from shim[0] (all cores share same A data)
            A_s[c] = object_fifo(f"AS{c}", st[0], mt[c], 2, A_l2_ty, None,
                [[(m, k), (mtk // k, m * k), (k, 1)]])
            A_c[c] = object_fifo(f"AC{c}", mt[c], ct[c], 2, A_l1_ty)
            object_fifo_link(A_s[c], A_c[c])
            
            # B from shim[c] (per N-block)
            B_s[c] = object_fifo(f"BS{c}", st[c], mt[c], 2, B_l2_ty)
            B_c[c] = object_fifo(f"BC{c}", mt[c], ct[c], 2, B_l1_ty)
            object_fifo_link(B_s[c], B_c[c])
            
            # C output
            C_c[c] = object_fifo(f"CC{c}", ct[c], mt[c], 1, C_l1_ty)
            C_s[c] = object_fifo(f"CS{c}", mt[c], st[c], 2, C_l2_ty)
            object_fifo_link(C_c[c], C_s[c])
        
        # Core body
        for c in range(n_aie_cols):
            @core(ct[c], stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
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
            n_mt = M // m
            n_nt = N // n
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
