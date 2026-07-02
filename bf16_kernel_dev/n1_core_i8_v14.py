#!/usr/bin/env python3
#
# INT8 MLIR generator v14 — TWO MEM TILES, ONE CORE
# - mem[0,1]: A fifo chain (shim[0] → mem[0,1] → core[0,2])
# - mem[1,1]: B fifo chain (shim[1] → mem[1,1] → core[1,2])
# - core[0,2]: processes M-tile
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
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = 512
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]  # 32x64 for m=128
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_128x64x128.o"  # DIM_M=128 kernel
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        s0 = tile(0, 0)
        mA = tile(0, 1)
        mB = tile(1, 1)
        c0 = tile(0, 2)
        s1 = tile(1, 0)
        
        # A on mem[0,1]
        A_L3L2 = object_fifo("A_L3L2", s0, mA, 2, A_l2_ty)
        A_L2L1 = object_fifo("A_L2L1", mA, c0, 2, A_l1_ty)
        object_fifo_link(A_L3L2, A_L2L1)
        
        # B on mem[1,1]
        B_L3L2 = object_fifo("B_L3L2", s1, mB, 2, B_l2_ty)
        B_L2L1 = object_fifo("B_L2L1", mB, c0, 2, B_l1_ty)
        object_fifo_link(B_L3L2, B_L2L1)
        
        # C out on mem[0,1]
        C_L1L2 = object_fifo("C_L1L2", c0, mA, 1, C_l1_ty)
        C_L2L3 = object_fifo("C_L2L3", mA, s0, 2, C_l2_ty)
        object_fifo_link(C_L1L2, C_L2L3)
        
        @core(c0)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range((M // m) * (N // n)):
                    C = C_L1L2.acquire(ObjectFifoPort.Produce, 1)
                    zero(C)
                    for _ in range_(K // k):
                        A = A_L2L1.acquire(ObjectFifoPort.Consume, 1)
                        B = B_L2L1.acquire(ObjectFifoPort.Consume, 1)
                        matmul(A, B, C)
                        A_L2L1.release(ObjectFifoPort.Consume, 1)
                        B_L2L1.release(ObjectFifoPort.Consume, 1)
                    C_L1L2.release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))
            for mi in range(M // m):
                for ni in range(N // n):
                    idx = mi * (N // n) + ni
                    at = shim_dma_single_bd_task(A_L3L2, A, tap=A_taps[mi], issue_token=False)
                    dma_start_task(at)
                    bt = shim_dma_single_bd_task(B_L3L2, B, tap=B_taps[ni], issue_token=False)
                    dma_start_task(bt)
                    ct = shim_dma_single_bd_task(C_L2L3, C, tap=C_taps[idx], issue_token=True)
                    dma_start_task(ct)
                    dma_await_task(at, bt, ct)
                    dma_free_task(at, bt, ct)

main()
