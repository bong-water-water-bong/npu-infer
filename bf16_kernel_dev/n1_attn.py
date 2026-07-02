#!/usr/bin/env python3
"""NPU attention xclbin generator. Single core, DMA fifos, Chess kernel.
   Q(16×128 bf16) + K(8×C×128 bf16) + V(8×C×128 bf16) → Out(16×128 bf16)
   C = KV context length (max 128, processed in 16-token blocks) """
import argparse, numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

NH=16; NKV=8; HD=128; GQA=NH//NKV  # 16 heads, GQA=2

def gen_attn(C=16):
    """Generate attention MLIR for context length C (max 128, processed in 16-token blocks).
       Single shim DMA + single core + single mem tile. Simple enough for aiecc. """
    Q_dwords = NH*HD//2     # 1024 i32 for 16×128 bf16
    K_dwords = NKV*C*HD//2  # up to 8192 i32 for 8×128×128 bf16
    V_dwords = K_dwords
    O_dwords = Q_dwords

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            # Buffers on core
            Q_ty = np.ndarray[(Q_dwords,), np.dtype[np.int32]]
            K_ty = np.ndarray[(K_dwords,), np.dtype[np.int32]]
            V_ty = np.ndarray[(V_dwords,), np.dtype[np.int32]]
            O_ty = np.ndarray[(O_dwords,), np.dtype[np.int32]]

            # Load our attention kernel (compiled with Peano)
            kernel_o = "attn_scalar.o"
            attn_fn = external_func("attn_scalar_bf16",
                inputs=[Q_ty, K_ty, V_ty, O_ty, np.int32],
                link_with=kernel_o)

            shim = tile(0, 0)
            mem = tile(0, 1)
            core = tile(0, 2)

            # ObjectFifos: Q in, K in, V in, O out — simple depth-2 fifos
            Q_fifo_shim = object_fifo("Q_shim", shim, mem, 2, Q_ty)
            Q_fifo_mem = object_fifo("Q_mem", mem, core, 2, Q_ty)
            object_fifo_link(Q_fifo_shim, Q_fifo_mem)

            K_fifo_shim = object_fifo("K_shim", shim, mem, 2, K_ty)
            K_fifo_mem = object_fifo("K_mem", mem, core, 2, K_ty)
            object_fifo_link(K_fifo_shim, K_fifo_mem)

            V_fifo_shim = object_fifo("V_shim", shim, mem, 2, V_ty)
            V_fifo_mem = object_fifo("V_mem", mem, core, 2, V_ty)
            object_fifo_link(V_fifo_shim, V_fifo_mem)

            O_fifo_mem = object_fifo("O_mem", core, mem, 1, O_ty)
            O_fifo_shim = object_fifo("O_shim", mem, shim, 2, O_ty)
            object_fifo_link(O_fifo_mem, O_fifo_shim)

            @core(core)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    Q = Q_fifo_mem.acquire(ObjectFifoPort.Consume, 1)
                    K = K_fifo_mem.acquire(ObjectFifoPort.Consume, 1)
                    V = V_fifo_mem.acquire(ObjectFifoPort.Consume, 1)
                    O = O_fifo_mem.acquire(ObjectFifoPort.Produce, 1)
                    attn_fn(Q, K, V, O, C)
                    Q_fifo_mem.release(ObjectFifoPort.Consume, 1)
                    K_fifo_mem.release(ObjectFifoPort.Consume, 1)
                    V_fifo_mem.release(ObjectFifoPort.Consume, 1)
                    O_fifo_mem.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(Q_dwords,), np.dtype[np.int32]],
                np.ndarray[(K_dwords,), np.dtype[np.int32]],
                np.ndarray[(V_dwords,), np.dtype[np.int32]],
                np.ndarray[(O_dwords,), np.dtype[np.int32]],
            )
            def sequence(Q_in, K_in, V_in, O_out):
                shim_dma_single_bd_task(Q_fifo_shim, Q_in, issue_token=True)
                shim_dma_single_bd_task(K_fifo_shim, K_in, issue_token=True)
                shim_dma_single_bd_task(V_fifo_shim, V_in, issue_token=True)
                t = shim_dma_single_bd_task(O_fifo_shim, O_out, issue_token=True)
                dma_await_task(t)

        print(ctx.module)

    return str(ctx.module)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-C", type=int, default=16)
    args = p.parse_args()
    gen_attn(args.C)
