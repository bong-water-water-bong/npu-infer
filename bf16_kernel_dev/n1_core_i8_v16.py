#!/usr/bin/env python3
#
# INT8 MLIR generator v16 — PER-CORE A FIFOS, mtk=k=64
# - mtk=64 = k: each A L2 element covers exactly ONE K-tile (32×64=2KB)
# - Per-core A fifos: each mem tile hosts its core's A + B + C
# - A per core: 2 elements × 2KB = 4KB (depth 2, or 1×2KB with depth 1)
# - B per core: 2 elements × 8KB = 16KB  
# - C per core: linked pool = max(1×8KB, 1×8KB) = 8KB
# - Total per mem tile: 4+16+8 = 28KB < 64KB ✅
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
    mtk = k  # = 64 — no K-sub-tiling, 1 K-tile per L2 element
    
    @device(AIEDevice.npu2)
    def device_body():
        # A_l2 = A_l1 = same type: 32×64 = 2KB
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
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
            # A: same type for L2 and L1 — single element = single K-tile
            A_s[c] = object_fifo(f"AS{c}", st[0], mt[c], 2, A_ty)
            A_c[c] = object_fifo(f"AC{c}", mt[c], ct[c], 2, A_ty)
            object_fifo_link(A_s[c], A_c[c])
            
            B_s[c] = object_fifo(f"BS{c}", st[c], mt[c], 2, B_l2_ty)
            B_c[c] = object_fifo(f"BC{c}", mt[c], ct[c], 2, B_l1_ty)
            object_fifo_link(B_s[c], B_c[c])
            
            C_c_f[c] = object_fifo(f"CC{c}", ct[c], mt[c], 1, C_l1_ty)
            C_s_f[c] = object_fifo(f"CS{c}", mt[c], st[c], 2, C_l2_ty)
            object_fifo_link(C_c_f[c], C_s_f[c])
        
        for c in range(n_aie_cols):
            @core(ct[c])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = C_c_f[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
                            Abuf = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c_f[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            # With no K-sub-tiling, each M-tile requires K//k elements from A
            # A_taps: one tap per M-tile, covering K-slices via repeat
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
                    bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[ni*n_aie_cols + c], issue_token=False)
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
