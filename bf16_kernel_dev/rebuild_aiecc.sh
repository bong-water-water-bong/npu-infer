#!/bin/bash
cd /home/bcloud/mlir-aie/build
ninja -j8 2>&1 | tail -10
echo "exit: $?"
