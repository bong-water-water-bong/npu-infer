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
def bf16_matmul(a: In, b: In, c: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int]):
    a_ty = np.ndarray[(M * K,), np.dtype[__import__('ml_dtypes').bfloat16]]
    b_ty = np.ndarray[(K * N,), np.dtype[__import__('ml_dtypes').bfloat16]]
    c_ty = np.ndarray[(M * N,), np.dtype[__import__('ml_dtypes').bfloat16]]
    
    of_a = ObjectFifo(a_ty, name='a')
    of_b = ObjectFifo(b_ty, name='b')
    of_c = ObjectFifo(c_ty, name='c')
    
    matmul = mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=__import__('ml_dtypes').bfloat16, output_dtype=__import__('ml_dtypes').bfloat16, use_chess=False)
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
print(f'Kernel _name: {mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=__import__("ml_dtypes").bfloat16, output_dtype=__import__("ml_dtypes").bfloat16, use_chess=False)._name}', flush=True)
print(f'mac_dims: {mm(dim_m=M, dim_k=K, dim_n=N, input_dtype=__import__("ml_dtypes").bfloat16, output_dtype=__import__("ml_dtypes").bfloat16, use_chess=False).mac_dims}', flush=True)

# Save the ELF before the pipeline cleans up
import os, shutil, struct, sys

# Run until failure, then save the .o
try:
    A = iron.tensor(np.zeros(M*K, dtype=np.uint16), dtype=np.uint16, device='npu')
    B = iron.tensor(np.zeros(K*N, dtype=np.uint16), dtype=np.uint16, device='npu')
    C = iron.tensor(np.zeros(M*N, dtype=np.uint16), dtype=np.uint16, device='npu')
    bf16_matmul(A, B, C, M=M, K=K, N=N)
except Exception as e:
    em = str(e)
    print(f'Error: {em[:200]}', flush=True)
    import re
    m = re.search(r'(\S+\.o)', em)
    if m:
        src = m.group(1)
        dst = '/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/peano_debug.o'
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'Saved ELF to {dst}', flush=True)
            # Inspect
            with open(dst, 'rb') as f:
                ident = f.read(16)
                elf_class = ident[4]
                print(f'ELF class: {elf_class} ({"32" if elf_class==1 else "64"})', flush=True)
            sys.exit(0)
    print('Could not save ELF', flush=True)
    sys.exit(1)
PYEOF
