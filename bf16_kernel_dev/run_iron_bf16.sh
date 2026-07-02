#!/bin/bash
set -e
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export PYTHONPATH=/home/bcloud/mlir-aie/build/python

cat > /tmp/iron_bf16_test2.py << 'PYEOF'
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import iron, Program, ObjectFifo, Worker, Runtime, In, Out
from aie.iron.kernels.linalg import mm

@iron.jit
def bf16_matmul(
    M: int, K: int, N: int,
    a_in: In, b_in: In, c_out: Out
):
    line_a_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    line_b_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    line_c_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    
    of_a = ObjectFifo(line_a_ty, name="a")
    of_b = ObjectFifo(line_b_ty, name="b")
    of_c = ObjectFifo(line_c_ty, name="c")
    
    matmul = mm(
        dim_m=M, dim_k=K, dim_n=N,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        use_chess=True
    )
    zero = matmul.zero
    
    def core(of_a, of_b, of_c, matmul, zero):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        zero(c)
        matmul(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)
    
    w = Worker(core, [of_a.cons(), of_b.cons(), of_c.prod(), matmul, zero])
    rt = Runtime()
    with rt.sequence(line_a_ty, line_b_ty, line_c_ty) as (a, b, c):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_b.prod(), b)
        rt.drain(of_c.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()

M, K, N = 128, 64, 128
np.random.seed(42)
A_np = np.random.rand(M, K).astype(np.float32)
B_np = np.random.rand(K, N).astype(np.float32)
C_ref = A_np @ B_np

A_iron = iron.array(A_np.ravel().view(np.uint16), shape=(M*K,), dtype=np.uint16, device="npu")
B_iron = iron.array(B_np.ravel().view(np.uint16), shape=(K*N,), dtype=np.uint16, device="npu")
C_iron = iron.zeros((M*N,), dtype=np.uint16, device="npu")

print(f"Running BF16 {M}x{K}x{N}...")
try:
    bf16_matmul(M, K, N, A_iron, B_iron, C_iron)
    C_out = C_iron.view(np.ndarray).view(np.float32).reshape(M, N)
    C_out = np.array(C_bf16, dtype=np.float32)
    err = np.max(np.abs(C_out - C_ref))
    print(f"Max error: {err:.4f}")
    print(f"MATCH: {err < 1.0}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
PYEOF

/home/bcloud/mlir-aie/.venv/bin/python /tmp/iron_bf16_test2.py 2>&1
