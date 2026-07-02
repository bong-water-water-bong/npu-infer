#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export PYTHONPATH=/home/bcloud/mlir-aie/build/python

/home/bcloud/mlir-aie/.venv/bin/python -c "
import aie.iron as iron
# Try to get current device
try:
    dev = iron.get_current_device()
    print('Device:', dev)
except Exception as e:
    print('get_current_device error:', e)

# Check what tensor class is available
from aie.iron import In, Out
print('In, Out types:', In, Out)
" 2>&1
