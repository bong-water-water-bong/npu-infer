#!/usr/bin/env python3
#
# INT8 MLIR generator v21 — DRAM-backed A fifo (same pattern as BFP16)
# - Uses --unified --dynamic-objFifos flag for large DRAM-backed pools
# - A_L3L2: 1 row, depth 2, element (m, mtk) int8 → m=128, mtk=512 → 64KB per element
# - A_L2L1: linked to 8 cores, depth 2, element (m//4, k) = (32, 64) = 2KB
# - Pool = max(2×64KB, 2×2KB) = 128KB → DRAM-backed
# - Pool has 128KB/2KB = 64 sub-views
# - Core loop: 4 inner A acquires per K-iteration, 16 K-iterations = 64 total = all sub-views once
# - Each sub-view covers unique (M-slice, K-slice) → NO K-interleaving!
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
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-mtk", type=int, default=512)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.mtk)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n, mtk):
    n_aie_cols = 8
    dtype_in = np.int8
    dtype_out = np.int16
    
    @device(AIEDevice.npu2)
    def device_body():
        # A types: DRAM-backed large element
        A_l2_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]  # 32×64 = 2048 = 2KB
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]  # 32×64 = 2048 = 2KB
        
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 64×128 = 8192 = 8KB
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 8KB
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 128×128×2 = 32KB
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 32KB
        
        kernel_o = "mm_128x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2]
        
        # A: DRAM-backed, linked pool with sub-views
        # dimensionsFromStream: [<size=m, stride=mtk>, <size=mtk//k, stride=m*k>, <size=k, stride=1>]
        # But in the bf16 design: dimensionsFromStream with complex sub-view pattern
        # The key: m_tile=128, k_sub=64, mtk=512
        
        # A_L3L2: shim[0] → mem[0], DRAM-backed element (m, mtk), depth 2
        # A_L3L2: same type as A_L1, no sub-view cycling needed
        A_L3L2 = object_fifo("A_L3L2", st[0], mt[0], 2, A_l2_ty, None)
        
        # A_L2L1: mem[0] → all 8 cores, depth 2
        # The shared A fifo links L2→L1. Link is automatic when types match
        # (A_L3L2 and A_L2L1 share the same mem tile).
        A_L2L1 = object_fifo("A_L2L1", mt[0], ct, 2, A_l1_ty)
        
        # B: per-column, as before
        B_s = [None]*n_aie_cols
        B_c = [None]*n_aie_cols
        C_c = [None]*n_aie_cols
        C_s = [None]*n_aie_cols
        
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"BS{c}", st[c], mt[c], 2, B_l2_ty)
            B_c[c] = object_fifo(f"BC{c}", mt[c], ct[c], 2, B_l1_ty)
            object_fifo_link(B_s[c], B_c[c])
            
            C_c[c] = object_fifo(f"CC{c}", ct[c], mt[c], 1, C_l1_ty)
            C_s[c] = object_fifo(f"CS{c}", mt[c], st[c], 2, C_l2_ty)
            object_fifo_link(C_c[c], C_s[c])
        
        for c in range(n_aie_cols):
            @core(ct[c])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            for _ in range_(m // (m // 4)):  # 4 = m/(m/4)
                                Abuf = A_L2L1.acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, Cbuf)
                                A_L2L1.release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            # A_taps: each tap covers (m, mtk) = (128, 512) = 64KB
            # With K=1024, mtk=512: 2 taps per M-group
            # num_row_tile = M // m // 1 = 1 (for M=128)
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))
            
            num_row_tile = M // m  # M/128
            num_col_tile = N // n // n_aie_cols  # N/1024
            num_groups = num_row_tile * num_col_tile
            tb = 4
            itg = [[] for _ in range(tb)]
            otg = [[] for _ in range(tb)]
            
            for gi in range(num_groups):
                a_idx = gi // num_col_tile
                for c in range(n_aie_cols):
                    bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[(gi % num_col_tile) * n_aie_cols + c], issue_token=False)
                    dma_start_task(bt); itg[gi%tb].append(bt)
                
                # A: shared, single BD for all cores
                at = shim_dma_single_bd_task(A_L3L2, A, tap=A_taps[a_idx], issue_token=False)
                dma_start_task(at); itg[gi%tb].append(at)
                
                for c in range(n_aie_cols):
                    ct_ = shim_dma_single_bd_task(C_s[c], C, tap=C_taps[gi * n_aie_cols + c], issue_token=True)
                    dma_start_task(ct_); otg[gi%tb].append(ct_)
                
                if gi%tb==1 and gi!=1:
                    for g in [2,3]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
                if gi%tb==3:
                    for g in [0,1]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
            for g in range(tb):
                if otg[g]: dma_await_task(*otg[g]); dma_free_task(*itg[g])

main()
