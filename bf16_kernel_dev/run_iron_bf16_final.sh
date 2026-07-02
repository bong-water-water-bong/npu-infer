#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python
export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH

/home/bcloud/mlir-aie/.venv/bin/python << 'PYEOF'
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, In, Out, CompileTime
from aie.iron.kernels.linalg import mm

@iron.jit
def bf16_matmul(a: In, b: In, c: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
    a_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    
    of_a = ObjectFifo(a_ty, name='a')
    of_b = ObjectFifo(b_ty, name='b')
    of_c = ObjectFifo(c_ty, name='c')
    
    matmul = mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=bfloat16, output_dtype=bfloat16, use_chess=False)
    zero = matmul.zero
    
    def core(of_a, of_b, of_c, matmul, zero):
        a_buf = of_a.acquire(1)
        b_buf = of_b.acquire(1)
        c_buf = of_c.acquire(1)
        zero(c_buf)
        matmul(a_buf, b_buf, c_buf)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)
    
    w = Worker(core, [of_a.cons(), of_b.cons(), of_c.prod(), matmul, zero])
    rt = Runtime()
    with rt.sequence(a_ty, b_ty, c_ty) as (ai, bi, ci):
        rt.start(w)
        rt.fill(of_a.prod(), ai)
        rt.fill(of_b.prod(), bi)
        rt.drain(of_c.cons(), ci, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()

M, K, N = 128, 64, 128
np.random.seed(42)
A_np = np.random.rand(M, K).astype(np.float32)
B_np = np.random.rand(K, N).astype(np.float32)
C_ref = A_np @ B_np

A = bfloat16(A_np.ravel())
B = bfloat16(B_np.ravel())
C = bfloat16(np.zeros(M * N, dtype=np.float32))

A_ir = iron.tensor(np.array(A, dtype=np.uint16).ravel(), dtype=np.uint16, device='npu')
B_ir = iron.tensor(np.array(B, dtype=np.uint16).ravel(), dtype=np.uint16, device='npu')
C_ir = iron.tensor(np.array(C, dtype=np.uint16).ravel(), dtype=np.uint16, device='npu')

print(f'Running BF16 {M}x{K}x{N} matmul via IRON...', flush=True)
bf16_matmul(A_ir, B_ir, C_ir, M=M, K=K, N=N)

C_out = np.frombuffer(C_ir.data, dtype=np.float32).reshape(M, N)
err = np.max(np.abs(C_out - C_ref))
print(f'Max error: {err:.4f}', flush=True)
print(f'PASS' if err < 1.0 else 'FAIL', flush=True)
PYEOF
