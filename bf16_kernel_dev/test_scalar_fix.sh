#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python

/home/bcloud/mlir-aie/.venv/bin/python -c "
from aie.extras.dialects.arith import ScalarValue, constant
from aie.ir import IndexType, Context
# Test creating a ScalarValue from an int literal
with Context():
    s = ScalarValue(42, dtype=IndexType.get())
print('ScalarValue created:', s)
print('literal:', s.literal_value)
print('SUCCESS')
" 2>&1
