#!/bin/bash
PYTHONPATH=/home/bcloud/mlir-aie/build/python LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64 /home/bcloud/mlir-aie/.venv/bin/python -c "
from ml_dtypes import bfloat16
import numpy as np
from aie.iron.kernels.linalg import mm

m = mm(dim_m=64, dim_k=64, dim_n=64, input_dtype=np.int8, output_dtype=np.int32, use_chess=False)
print('name:', m._name)
print('original:', m._original_name)
print('zero name:', m.zero._name if hasattr(m, 'zero') else 'N/A')
print('arg_types:', [str(t) for t in m._arg_types])
" > /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/mm_info.txt 2>&1
cat /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/mm_info.txt
