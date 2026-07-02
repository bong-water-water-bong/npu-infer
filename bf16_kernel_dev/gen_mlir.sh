#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
/tmp/mlir_venv/bin/python3 gen_mlir_v2.py 2>&1
