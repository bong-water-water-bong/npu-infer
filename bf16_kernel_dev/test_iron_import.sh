#!/bin/bash
cd /tmp
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs:$LD_LIBRARY_PATH
PYTHONPATH=/home/bcloud/mlir-aie/build/python \
  /home/bcloud/mlir-aie/.venv/bin/python -c "
import sys
sys.path.insert(0, '/home/bcloud/mlir-aie/build/python')
sys.path.insert(0, '/home/bcloud/torch2aie/toolchain/mlir_aie/python')
try:
    from ml_dtypes import bfloat16
    from aie.iron.kernels.linalg import mm
    e = mm(dim_m=128, dim_k=64, dim_n=128, input_dtype=bfloat16, output_dtype=bfloat16, use_chess=True)
    print('BF16 mac_dims:', e.mac_dims)
    
    e2 = mm(dim_m=128, dim_k=64, dim_n=128, input_dtype=__import__('numpy').int8, output_dtype=__import__('numpy').int8, use_chess=True)
    print('INT8 mac_dims:', e2.mac_dims)
except Exception as ex:
    print('ERROR:', ex)
    import traceback
    traceback.print_exc()
" 2>&1
