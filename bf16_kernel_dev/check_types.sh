#!/bin/bash
PYTHONPATH=/home/bcloud/torch2aie/toolchain/mlir_aie/python /home/bcloud/torch2aie/.venv/bin/python -c "
from aie.dialects.aiex import *
import sys
names = [x for x in dir() if 'int' in x.lower() or 'bf' in x.lower()]
names.sort()
sys.stdout.write('Found types: ' + ', '.join(names[:30]) + '\n')
" 2>&1
