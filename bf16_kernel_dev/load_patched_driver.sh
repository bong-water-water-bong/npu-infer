#!/bin/bash
set -e
echo "=== Removing old module ==="
sudo rmmod amdxdna 2>&1 || true
echo "=== Inserting patched module ==="
sudo insmod /home/bcloud/amdxdna-dkms/src/amdxdna/amdxdna.ko 2>&1
echo "=== Module loaded ==="
lsmod | grep amdxdna
echo "=== Testing 12-col xclbin ==="
LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  /home/bcloud/npu-sandbox/npu-infer/build/test_12col 2>&1
echo "Test exit: $?"
