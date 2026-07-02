#!/usr/bin/env python3
#
# INT8 MLIR generator v18 — PER-CORE A FIFOS, UNLINKED
# - A_s[c] and A_c[c] are NOT linked — separate buffer pools
# - A_s fills L2 buffer from shim DMA (DDR→L2)
# - A_c reads from L2 buffer to L1 (L2→core) with its OWN BD
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
    mtk = k  # NO K-sub-tiling: each L2 element = 1 K-tile
    
    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]  # 32×64 = 2KB
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]] 
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2]
        
        A_s = [None]*n_aie_cols
        A_c = [None]*n_aie_cols
        B_s = [None]*n_aie_cols
        B_c = [None]*n_aie_cols
        C_c_f = [None]*n_aie_cols
        C_s_f = [None]*n_aie_cols
        
        for c in range(n_aie_cols):
            # A: shim[c] → mem[c] (2KB elements, depth=K//k for full K coverage)
            A_s[c] = object_fifo(f"AS{c}", st[c], mt[c], K//k, A_ty)  # depth=16!
            # A: mem[c] → core[c] — NOT linked, separate fifo
            A_c[c] = object_fifo(f"AC{c}", mt[c], ct[c], 1, A_ty)  # depth=1
            
            B_s[c] = object_fifo(f"BS{c}", st[c], mt[c], K//k, B_l2_ty)
            B_c[c] = object_fifo(f"BC{c}", mt[c], ct[c], 1, B_l1_ty)
            
            C_c_f[c] = object_fifo(f"CC{c}", ct[c], mt[c], 1, C_l1_ty)
            C_s_f[c] = object_fifo(f"CS{c}", mt[c], st[c], 2, C_l2_ty)
        
        for c in range(n_aie_cols):
            @core(ct[c])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = C_c_f[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
                            A = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(A, B_c[c].acquire(ObjectFifoPort.Consume, 1), Cbuf)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c_f[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))
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
                for c in range(n_aie_cols):
                    ct_ = shim_dma_single_bd_task(C_s_f[c], C, tap=C_taps[gi*n_aie_cols + c], issue_token=True)
                    dma_start_task(ct_); otg[gi%tb].append(ct_)
                if gi%tb==1 and gi!=1:
                    for g in [2,3]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
                if gi%tb==3:
                    for g in [0,1]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
            for g in range(tb):
                if otg[g]: dma_await_task(*otg[g]); dma_free_task(*itg[g])

main()
