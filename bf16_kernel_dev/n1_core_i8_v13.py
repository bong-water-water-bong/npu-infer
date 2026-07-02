#!/usr/bin/env python3
#
# INT8 MLIR generator v13 — SINGLE CORE, ALL FIFOS DEDICATED
# - 1 core only (no sharing, no K-slice interleaving)
# - Each mem tile handles ONE fifo chain
# - Simple memory layout
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
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_cores = 1  # Single core
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = 512
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        st0 = tile(0, 0)
        mt0 = tile(0, 1)
        ct0 = tile(0, 2)
        
        A_l3l2 = object_fifo("A_L3L2", st0, mt0, 2, A_l2_ty)
        A_l2l1 = object_fifo("A_L2L1", mt0, ct0, 2, A_l1_ty)
        object_fifo_link(A_l3l2, A_l2l1)
        
        B_l3l2 = object_fifo("B_L3L2", st0, mt0, 2, B_l2_ty)
        B_l2l1 = object_fifo("B_L2L1", mt0, ct0, 2, B_l1_ty)
        object_fifo_link(B_l3l2, B_l2l1)
        
        C_l1l2 = object_fifo("C_L1L2", ct0, mt0, 1, C_l1_ty)
        C_l2l3 = object_fifo("C_L2L3", mt0, st0, 2, C_l2_ty)
        object_fifo_link(C_l1l2, C_l2l3)
        
        @core(ct0)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range((M // m) * (N // n) // n_cores):
                    C = C_l1l2.acquire(ObjectFifoPort.Produce, 1)
                    zero(C)
                    for _ in range_(K // k):
                        A = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                        B = B_l2l1.acquire(ObjectFifoPort.Consume, 1)
                        matmul(A, B, C)
                        A_l2l1.release(ObjectFifoPort.Consume, 1)
                        B_l2l1.release(ObjectFifoPort.Consume, 1)
                    C_l1l2.release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))
            n_mt = M // m
            n_nt = N // n
            for mi in range(n_mt):
                for ni in range(n_nt):
                    idx = mi * n_nt + ni
                    a_task = shim_dma_single_bd_task(A_l3l2, A, tap=A_taps[mi], issue_token=False)
                    dma_start_task(a_task)
                    b_task = shim_dma_single_bd_task(B_l3l2, B, tap=B_taps[ni], issue_token=False)
                    dma_start_task(b_task)
                    c_task = shim_dma_single_bd_task(C_l2l3, C, tap=C_taps[idx], issue_token=True)
                    dma_start_task(c_task)
                    dma_await_task(a_task, b_task, c_task)
                    dma_free_task(a_task, b_task, c_task)

main()
