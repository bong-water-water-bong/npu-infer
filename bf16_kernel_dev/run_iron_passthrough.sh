#!/bin/bash
set -e
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python

cat > /tmp/iron_passthrough.py << 'PYEOF'
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, In, Out, CompileTime
import aie.iron.kernels as kernels

@iron.jit
def passthrough_design(a: In, b: Out, *, N: CompileTime[int]):
    line_ty = np.ndarray[(N,), np.dtype[np.uint8]]
    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")
    
    pt = kernels.passthrough(tile_size=N, dtype=np.uint8)
    
    def core_fn(of_in, of_out, pt):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        pt(elem_in, elem_out, N)
        of_in.release(1)
        of_out.release(1)
    
    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), pt])
    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()

N = 1024
a = iron.arange(N, dtype=np.uint8, device="npu")
b = iron.zeros(N, dtype=np.uint8, device="npu")
print("Running passthrough...")
passthrough_design(a, b, N=N)  # N is CompileTime, passed as kwarg
print("Done!")
print("First 8:", np.array(b)[:8])
PYEOF

/home/bcloud/mlir-aie/.venv/bin/python /tmp/iron_passthrough.py 2>&1
