#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Test IRON API BF16 matmul
export PYTHONPATH="/home/bcloud/mlir-aie/npu2_40_toolchain/python:/home/bcloud/torch2aie/toolchain/mlir_aie/python:$PYTHONPATH"
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export LD_LIBRARY_PATH="/home/bcloud/mlir-aie/npu2_40_toolchain/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs:$LD_LIBRARY_PATH"

/home/bcloud/torch2aie/.venv/bin/python -c "
from aie.iron.kernels.linalg import mm
from ml_dtypes import bfloat16
import numpy as np

# Check the mm function
print('MM function:', mm)
print('mac_dims:', mm(dim_m=128, dim_k=64, dim_n=128, input_dtype=bfloat16, output_dtype=bfloat16, use_chess=True).mac_dims)
print('IRON API available and BF16 supported!')
" 2>&1
