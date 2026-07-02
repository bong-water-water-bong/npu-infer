#!/usr/bin/env python3
#
# INT8 MLIR generator v20 — PER-SHIM A FIFOS (unlinked)
# Each shim tile → mem tile → core tile a SEPARATE (unlinked) objectfifo
# No dimensionsFromStream — mtk = k = 64, no K-sub-views
# Depth = K//k = 16 for A fifo
# B depth = 2 (reduced from K//k)
# C depth = 1
#
# With per-shim distribution, each core's K-loop gets ALL 16 K-slices
# via its own A fifo (no K-interleaving across cores)
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
    dtype_in = np.int8
    dtype_out = np.int16
    
    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]  # 32×64i8 = 2KB
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 64×128i8 = 8KB
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # 64×128i8 = 8KB
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 32×128i16 = 8KB
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # 32×128i16 = 8KB
        
        # A depth = K//k = 16 (one per K-slice) — UNLINKED so no linked-pool limit
        # The fifo goes shim→mem (depth=K//k) and mem→core (depth=1) UNLINKED
        # Actually if NOT linked, we have separate pools
        # AS: depth=K//k, element=2KB → pool=32KB (fits in mem 64KB)
        # AC: depth=1, element=2KB → pool=2KB
        # Total mem per column: A=32KB + B=8KB + C=8KB = 48KB ✅
        
        a_depth = K // k  # 16
        b_depth = 2
        c_depth = 1
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        st, mt, ct = tiles[0], tiles[1], tiles[2]
        
        AS_fifos = []
        AC_fifos = []
        BS_fifos = []
        BC_fifos = []
        CC_fifos = []
        CS_fifos = []
        
        for c in range(n_aie_cols):
            # Per-column A fifos: UNLINKED
            AS = object_fifo(f"AS{c}", st[c], mt[c], a_depth, A_ty)
            AC = object_fifo(f"AC{c}", mt[c], ct[c], 1, A_ty)
            AS_fifos.append(AS)
            AC_fifos.append(AC)
            
            BS = object_fifo(f"BS{c}", st[c], mt[c], b_depth, B_l2_ty)
            BC = object_fifo(f"BC{c}", mt[c], ct[c], 1, B_l1_ty)
            object_fifo_link(BS, BC)
            BS_fifos.append(BS)
            BC_fifos.append(BC)
            
            CC = object_fifo(f"CC{c}", ct[c], mt[c], c_depth, C_l1_ty)
            CS = object_fifo(f"CS{c}", mt[c], st[c], c_depth, C_l2_ty)
            object_fifo_link(CC, CS)
            CC_fifos.append(CC)
            CS_fifos.append(CS)
        
        for c in range(n_aie_cols):
            @core(ct[c])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range((M // m) * (N // n) // n_aie_cols):
                        Cbuf = CC_fifos[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(K // k):
                            Abuf = AC_fifos[c].acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = BC_fifos[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            AC_fifos[c].release(ObjectFifoPort.Consume, 1)
                            BC_fifos[c].release(ObjectFifoPort.Consume, 1)
                        CC_fifos[c].release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            # A: per-column tap, each reads the SAME A data
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
                    # Each shim reads the SAME A tap
                    at = shim_dma_single_bd_task(AS_fifos[c], A, tap=A_taps[mi], issue_token=False)
                    dma_start_task(at); itg[gi%tb].append(at)
                    bt = shim_dma_single_bd_task(BS_fifos[c], B, tap=B_taps[ni*n_aie_cols + c], issue_token=False)
                    dma_start_task(bt); itg[gi%tb].append(bt)
                for c in range(n_aie_cols):
                    ct_ = shim_dma_single_bd_task(CS_fifos[c], C, tap=C_taps[gi*n_aie_cols + c], issue_token=True)
                    dma_start_task(ct_); otg[gi%tb].append(ct_)
                if gi%tb==1 and gi!=1:
                    for g in [2,3]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
                if gi%tb==3:
                    for g in [0,1]: dma_await_task(*otg[g]); dma_free_task(*itg[g]); otg[g]=[]; itg[g]=[]
            for g in range(tb):
                if otg[g]: dma_await_task(*otg[g]); dma_free_task(*itg[g])

main()
