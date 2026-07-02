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
    
    matmul = mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=bfloat16, output_dtype=bfloat16, use_chess=False, vectorized=True)
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

M, K, N = 32, 32, 32  # Smaller for BF16 (larger element size)
np.random.seed(42)
A_np = np.random.rand(M, K).astype(np.float32)
B_np = np.random.rand(K, N).astype(np.float32)
C_ref = A_np @ B_np

def f32_to_u16_bf16(f):
    v = np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0]
    r = ((v >> 16) & 1) + 0x7FFF
    return np.uint16((v + r) >> 16)

# Vectorized kernel uses column-major B and C (default b_col_maj=False)
# Transpose B row-major → column-major on host
A_bf16 = np.vectorize(f32_to_u16_bf16, otypes=[np.uint16])(A_np.ravel())
B_bf16 = np.vectorize(f32_to_u16_bf16, otypes=[np.uint16])(B_np.T.ravel())  # column-major
C_bf16 = np.zeros(M * N, dtype=np.uint16)

A_ir = iron.tensor(A_bf16, dtype=np.uint16, device='npu')
B_ir = iron.tensor(B_bf16, dtype=np.uint16, device='npu')
C_ir = iron.tensor(C_bf16, dtype=np.uint16, device='npu')

print(f'BF16 {M}x{K}x{N} matmul via IRON...', flush=True)
bf16_matmul(A_ir, B_ir, C_ir, M=M, K=K, N=N)

C_raw = np.array(C_ir, dtype=np.uint16)
# C is column-major from kernel. Convert to row-major: C_rm[r,c] = C_cm[r + c*M]
def u16_to_f32_bf16(v):
    return np.frombuffer(np.uint32(v << 16).tobytes(), dtype=np.float32)[0]
C_cm = np.vectorize(u16_to_f32_bf16, otypes=[np.float32])(C_raw)
C_out = C_cm.reshape(N, M).T  # column-major → row-major
err = np.max(np.abs(C_out - C_ref))
print(f'Max error: {err:.6f}', flush=True)
print(f'Sample NPU: {C_out[0,:4]}', flush=True)
print(f'Sample CPU: {C_ref[0,:4]}', flush=True)
print('PASS' if err < 0.1 else 'FAIL', flush=True)
PYEOF
