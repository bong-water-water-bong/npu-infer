#!/bin/bash
# Check symbols in the last compiled kernel object
find /home/bcloud/.npu/cache -name "*.o" -type f 2>/dev/null | head -3
