#!/bin/bash
export LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs

g++ -std=c++23 -O2 -o /home/bcloud/npu-sandbox/npu-infer/build/test_i8_gemm /home/bcloud/npu-sandbox/npu-infer/tools/test_mt_gemm.cpp \
  -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -L/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -Wl,-rpath,/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -Wl,-rpath,/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -lxrt_coreutil -luuid -ldl 2>&1
echo "Build: $?"
