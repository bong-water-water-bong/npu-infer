#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64
export PYTHONPATH=/home/bcloud/mlir-aie/build/python
/home/bcloud/mlir-aie/.venv/bin/python -c "
import sys
for p in sys.path: print(p)
" > /tmp/iron_test_out.txt 2>&1
cat /tmp/iron_test_out.txt
