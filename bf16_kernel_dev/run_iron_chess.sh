#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python
export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH

/home/bcloud/mlir-aie/.venv/bin/python << 'PYEOF'
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, In, Out, CompileTime
from aie.iron.kernels.eltwise import passthrough

@iron.jit
def pt_design(a: In, b: Out, *, N: CompileTime[int]):
    line_ty = np.ndarray[(N,), np.dtype[np.uint8]]
    of_in = ObjectFifo(line_ty, name='in')
    of_out = ObjectFifo(line_ty, name='out')
    # Use Peano compiler with no prefix (avoids symbol rename entirely)
    from aie.iron.kernel import ExternalFunction
    tile_ty = np.ndarray[(N,), np.dtype[np.uint8]]
    kp = ExternalFunction(
        "passThroughLine",
        source_file="/home/bcloud/mlir-aie/aie_kernels/generic/passThrough.cc",
        arg_types=[tile_ty, tile_ty, np.int32],
        compile_flags=["-DBIT_WIDTH=8"],
        use_chess=False,
        include_dirs=["/home/bcloud/mlir-aie/build/include"],
    )
    
    def core(of_in, of_out, kp):
        ei = of_in.acquire(1)
        eo = of_out.acquire(1)
        kp(ei, eo, N)
        of_in.release(1)
        of_out.release(1)
    w = Worker(core, [of_in.cons(), of_out.prod(), kp])
    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (ai, bo):
        rt.start(w)
        rt.fill(of_in.prod(), ai)
        rt.drain(of_out.cons(), bo, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()

N = 1024
a = iron.arange(N, dtype=np.uint8, device='npu')
b = iron.zeros(N, dtype=np.uint8, device='npu')
print('Running IRON passthrough (chess)...')
pt_design(a, b, N=N)
print('Success! First 8:', np.array(b)[:8])
PYEOF
