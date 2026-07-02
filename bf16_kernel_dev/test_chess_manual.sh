#!/bin/bash
# Test Chess kernel compilation the way config1 Makefile does it
export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH
export AIETOOLS=/home/bcloud/torch2aie/toolchain/aietools
export AIETOOLS_DIR=$AIETOOLS

KERNEL_SRC=/home/bcloud/mlir-aie/aie_kernels/generic/passThrough.cc
OUT_DIR=/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build
mkdir -p "$OUT_DIR"

echo "Compiling passThrough.cc with Chess..."
xchesscc_wrapper aie2p -c "$KERNEL_SRC" -o "$OUT_DIR/passThrough_test.o" 2>&1
echo "Exit: $?"
ls -la "$OUT_DIR/passThrough_test.o" 2>/dev/null
