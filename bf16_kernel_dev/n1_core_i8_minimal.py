#!/usr/bin/env python3
#
# MINIMAL INT8 MLIR generator for AMD NPU2
# Single-core, single-tile: 32x64x128 matmul
# A: memref<32x64xi8>, B: memref<64x128xi8>, C: memref<128x128xi16>
# Kernel: mm_32x64x128.o with DIM_M=32
#
# This tests the simplest possible INT8 xclbin.
import argparse
from ml_dtypes import bfloat16
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=32)
    parser.add_argument("-K", type=int, default=64)
    parser.add_argument("-N", type=int, default=128)
    args = parser.parse_args()
    
    M, K, N = args.M, args.K, args.N
    
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            # Single tile types
            a_ty = np.ndarray[(M, K), np.dtype[np.int8]]
            b_ty = np.ndarray[(K, N), np.dtype[np.int8]]
            c_ty = np.ndarray[(M, N), np.dtype[np.int16]]
            
            # External kernel
            matmul = external_func(
                "matmul_i8_i16",
                inputs=[a_ty, b_ty, c_ty],
                link_with="mm_32x64x128.o",
            )
            zero = external_func(
                "zero_i16",
                inputs=[c_ty],
                link_with="mm_32x64x128.o",
            )
            
            # Single shim tile, mem tile, core tile
            st = tile(0, 0)
            mt = tile(0, 1)
            ct = tile(0, 2)
            
            # ObjectFifos
            # A: shim → mem → core
            of_a_s = object_fifo("a_s", st, mt, 2, a_ty)
            of_a_c = object_fifo("a_c", mt, ct, 2, a_ty)
            object_fifo_link(of_a_s, of_a_c)
            # B: shim → mem → core
            of_b_s = object_fifo("b_s", st, mt, 2, b_ty)
            of_b_c = object_fifo("b_c", mt, ct, 2, b_ty)
            object_fifo_link(of_b_s, of_b_c)
            # C: core → mem → shim
            of_c_c = object_fifo("c_c", ct, mt, 1, c_ty)
            of_c_s = object_fifo("c_s", mt, st, 2, c_ty)
            object_fifo_link(of_c_c, of_c_s)
            
            # Core body: zero C, matmul(A,B,C)
            @core(ct)
            def core_body():
                elem_out = of_c_c.acquire(ObjectFifoPort.Produce, 1)
                zero(elem_out)
                elem_in_a = of_a_c.acquire(ObjectFifoPort.Consume, 1)
                elem_in_b = of_b_c.acquire(ObjectFifoPort.Consume, 1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a_c.release(ObjectFifoPort.Consume, 1)
                of_b_c.release(ObjectFifoPort.Consume, 1)
                of_c_c.release(ObjectFifoPort.Produce, 1)
            
            # Runtime sequence
            @runtime_sequence(
                np.ndarray[(M * K,), np.dtype[np.int8]],
                np.ndarray[(K * N,), np.dtype[np.int8]],
                np.ndarray[(M * N,), np.dtype[np.int16]],
            )
            def sequence(A, B, C):
                a_task = shim_dma_single_bd_task(of_a_s, A, issue_token=False)
                b_task = shim_dma_single_bd_task(of_b_s, B, issue_token=False)
                c_task = shim_dma_single_bd_task(of_c_s, C, issue_token=True)
                dma_start_task(a_task)
                dma_start_task(b_task)
                dma_start_task(c_task)
                dma_await_task(c_task)
                dma_free_task(a_task)
                dma_free_task(b_task)
                dma_free_task(c_task)
        
        print(ctx.module)


main()
