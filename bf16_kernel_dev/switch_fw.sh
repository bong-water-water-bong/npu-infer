#!/bin/bash
# Test older firmware for >8 column support
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FW_DIR="/home/bcloud/amdxdna-dkms/firmware/amdnpu/17f0_11"
CURRENT="$FW_DIR/npu.sbin.1.1.2.65"
OLDER="$FW_DIR/npu.sbin.1.0.0.166"

echo "Current firmware: $(stat -c%s $CURRENT) bytes"
echo "Older firmware: $(stat -c%s $OLDER) bytes"

# Backup current symlink
sudo mv "$FW_DIR/npu.sbin" "$FW_DIR/npu.sbin.backup" 2>/dev/null || true

# Switch to older firmware
echo "Switching to older firmware..."
sudo ln -sf "npu.sbin.1.0.0.166" "$FW_DIR/npu.sbin"
echo "Older firmware linked. Checking..."
ls -la "$FW_DIR/npu.sbin"

# Tell the user to reboot or reload the driver
echo ""
echo "Firmware switched. Need to reload amdxdna driver."
echo "Options:"
echo "  sudo rmmod amdxdna && sudo modprobe amdxdna"
echo "  OR reboot"
echo ""
echo "Then test with:"
echo "  bash /home/bcloud/npu-sandbox/npu-infer/build/run_fused.sh"
