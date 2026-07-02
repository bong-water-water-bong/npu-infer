#!/usr/bin/env python3
#
# INT8 MLIR generator v19 — Shared A fifo with DEPTH 16
# - A_L3L2 from shim[0] to mem[0,1], depth = K//mtk * mtk//k = 2 * 8 = 16
# - A_L2L1 from mem[0,1] to 8 cores, depth = 16 (matches pool size)
# - Pool = 16 * 2KB = 32KB... wait, A_L3L2 element = 32×512 = 16KB
# - With depth 16: 16×16KB = 256KB > mem tile limits!
#
# Alternative: mtk = k = 64, no K-sub-tiling
# - A_L3L2 element = 32×64 = 2KB
# - A_L2L1 element = 32×64 = 2KB  
# - Depth = K//k = 16 for both
# - Pool = max(16×2KB, 16×2KB) = 32KB (fits in 64KB!)
# - Each core gets unique K-sub-view: core 0 gets K[0:63], core 1 K[64:127], etc.
# - With depth 16 (== number of sub-views): each acquire gives next sub-view
#
# Memory per mem tile:
# - A_L3L2: 16 × 2KB = 32KB (A buffer in mem tile)
# - B fifos: 2 × 8KB = 16KB
# - C fifos: 2 × 8KB = 16KB
# Total: 64KB — AT THE LIMIT, might fail
#
# Make B fifo depth 1? B element = 64×128×1 = 8192 bytes
# B depth 1: 1×8KB = 8KB
# C depth 1: 1×8KB = 8KB (output)
# Total: 32KB + 8KB + 8KB = 48KB ✅ (fits!)
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
        # Element = full K-tile (no K-sub-tiling)
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]  # 32×64 = 2KB
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 64×128 = 8KB
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 8KB
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 32×128×2 = 8KB
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 8KB
        
        # A depth = K//k = 16 (one element per K-slice)
        a_depth = K // k
        # B depth = 1 (reduces memory)
        b_depth = 1
        # C depth = 1 (output, 8KB)
        c_depth = 1
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2]
        
        sA = st[0]
        mA = mt[0]
        
        # Shared A: depth = a_depth (16)
        # shim[0] → mem[0,1] → all cores
        A_s = object_fifo("A_L3L2", sA, mA, a_depth, A_ty)
        A_c = object_fifo("A_L2L1", mA, ct, a_depth, A_ty)
        object_fifo_link(A_s, A_c)
        
        # Per-core B and C
        B_s = [None]*n_aie_cols
        B_c = [None]*n_aie_cols
        C_c_f = [None]*n_aie_cols
        C_s_f = [None]*n_aie_cols
        
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"BS{c}", st[c], mt[c], b_depth, B_l2_ty)
            B_c[c] = object_fifo(f"BC{c}", mt[c], ct[c], 1, B_l1_ty)
            object_fifo_link(B_s[c], B_c[c])
            
            C_c_f[c] = object_fifo(f"CC{c}", ct[c], mt[c], c_depth, C_l1_ty)
            C_s_f[c] = object_fifo(f"CS{c}", mt[c], st[c], c_depth, C_l2_ty)
            object_fifo_link(C_c_f[c], C_s_f[c])
        
        for c in range(n_aie_cols):
            @core(ct[c])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = C_c_f[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
                            A = A_c.acquire(ObjectFifoPort.Consume, 1)
                            matmul(A, B_c[c].acquire(ObjectFifoPort.Consume, 1), Cbuf)
                            A_c.release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c_f[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            # A: no K-sub-tiling, each tile is (m, k) = (32, 64)
            # With mtk=k=64, each tap covers 32×K elements
            # group_tiler: (M, K), (m, k), step (1, K//k) = (1, 16)
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
                # A: single BD for all cores, depth 16 (one per K-slice)
                at = shim_dma_single_bd_task(A_s, A, tap=A_taps[mi], issue_token=False)
                dma_start_task(at); itg[gi%tb].append(at)
                for c in range(n_aie_cols):
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
