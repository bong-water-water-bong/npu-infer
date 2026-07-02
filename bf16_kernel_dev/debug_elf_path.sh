#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export PYTHONPATH=/home/bcloud/mlir-aie/build/python:/home/bcloud/torch2aie/toolchain/xrt/python
export PATH=/home/bcloud/torch2aie/toolchain/bin:$PATH
PY=/home/bcloud/mlir-aie/.venv/bin/python

$PY /tmp/iron_passthrough_final.py 2>&1

# Now check for the .o file in cache
ls /home/bcloud/.npu/cache/ 2>/dev/null
