#!/bin/bash
set -e
cd /home/bcloud/amdxdna-dkms/src/amdxdna
make -C /lib/modules/$(uname -r)/build M=$PWD modules 2>&1 | tail -10
echo "Build exit: $?"
ls -la amdxdna.ko
