#!/bin/bash
LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs /home/bcloud/npu-sandbox/npu-infer/build/test_12col 2>&1
echo "Exit: $?"
