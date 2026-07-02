#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python
export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH

/home/bcloud/mlir-aie/.venv/bin/python << 'PYEOF'
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, In, Out, CompileTime
from aie.iron.kernels.linalg import mm

@iron.jit
def i8_matmul(a: In, b: In, c: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
    a_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[(K * N,), np.dtype[np.int8]]
    c_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    
    of_a = ObjectFifo(a_ty, name='a')
    of_b = ObjectFifo(b_ty, name='b')
    of_c = ObjectFifo(c_ty, name='c')
    
    matmul = mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=np.int8, output_dtype=np.int32, use_chess=False, vectorized=False)
    zero = matmul.zero
    
    def core(of_a, of_b, of_c, matmul):
        a_buf = of_a.acquire(1)
        b_buf = of_b.acquire(1)
        c_buf = of_c.acquire(1)
        # C is zeroed by host (iron.zeros), skip zero kernel call
        matmul(a_buf, b_buf, c_buf)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)
    
    w = Worker(core, [of_a.cons(), of_b.cons(), of_c.prod(), matmul])
    rt = Runtime()
    with rt.sequence(a_ty, b_ty, c_ty) as (ai, bi, ci):
        rt.start(w)
        rt.fill(of_a.prod(), ai)
        rt.fill(of_b.prod(), bi)
        rt.drain(of_c.cons(), ci, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()

M, K, N = 64, 64, 64
np.random.seed(42)
A_np = np.random.randint(-10, 10, (M, K)).astype(np.int8)
B_np = np.random.randint(-10, 10, (K, N)).astype(np.int8)
C_ref = A_np.astype(np.int32) @ B_np.astype(np.int32)

# Kernel now uses row-major B and C (b_col_maj=True, c_col_maj=True)
A_ir = iron.tensor(np.ascontiguousarray(A_np.ravel()), dtype=np.int8, device='npu')
B_ir = iron.tensor(np.ascontiguousarray(B_np.ravel()), dtype=np.int8, device='npu')
C_ir = iron.zeros(M * N, dtype=np.int32, device='npu')

print(f'INT8 {M}x{K}x{N} matmul via IRON...', flush=True)
i8_matmul(A_ir, B_ir, C_ir, M=M, K=K, N=N)

C_raw = np.array(C_ir, dtype=np.int32)
# Kernel outputs row-major C (c_col_maj=True)
C_out = C_raw.reshape(M, N)
err = np.max(np.abs(C_out.astype(np.float64) - C_ref.astype(np.float64)))
print(f'Max error: {err:.4f}', flush=True)
print(f'Sample NPU: {C_out[0,:4]}', flush=True)
print(f'Sample CPU: {C_ref[0,:4]}', flush=True)
print('PASS' if err < 2 else 'FAIL', flush=True)
PYEOF
