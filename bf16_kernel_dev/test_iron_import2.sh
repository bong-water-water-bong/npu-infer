#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64
export PYTHONPATH=/home/bcloud/mlir-aie/build/python
/home/bcloud/mlir-aie/.venv/bin/python -c "
from ml_dtypes import bfloat16
from aie.iron.kernels.linalg import mm
e = mm(dim_m=128, dim_k=64, dim_n=128, input_dtype=bfloat16, output_dtype=bfloat16, use_chess=True)
print('BF16 mac_dims:', e.mac_dims)
e2 = mm(dim_m=128, dim_k=64, dim_n=128, input_dtype=__import__('numpy').int8, output_dtype=__import__('numpy').int8, use_chess=True)
print('INT8 mac_dims:', e2.mac_dims)
print('SUCCESS: IRON API works!')
" > /tmp/iron_test_out.txt 2>&1
cat /tmp/iron_test_out.txt
