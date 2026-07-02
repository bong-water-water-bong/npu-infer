#!/bin/bash
# Build ALL INT8 xclbins for all HD=128 compatible models in one shot
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== Building all model xclbins ==="
echo ""

echo "1/5: Qwen3-0.6B  (4 xclbins, GU combined)"
bash "$SCRIPT_DIR/build_qwen3_0_6b_xclbins.sh" || echo "⚠️ Qwen3-0.6B failed"

echo ""
echo "2/5: Qwen3-VL-4B (5 xclbins, GU split)"
bash "$SCRIPT_DIR/build_qwen3_vl_xclbins.sh" || echo "⚠️ Qwen3-VL-4B failed"

echo ""
echo "3/5: Qwen3-8B    (5 xclbins, GU split)"
bash "$SCRIPT_DIR/build_qwen3_8b_xclbins.sh" || echo "⚠️ Qwen3-8B failed"

echo ""
echo "4/5: Llama-3.1-8B (5 xclbins, GU split)"
bash "$SCRIPT_DIR/build_llama_xclbins.sh" || echo "⚠️ Llama-3.1-8B failed"

echo ""
echo "5/5: Gemma4-E2B  (4 xclbins, GU combined)"
bash "$SCRIPT_DIR/build_gemma4_xclbins.sh" || echo "⚠️ Gemma4-E2B failed"

echo ""
echo "=== ALL DONE ==="
echo ""
# Count tagged xclbins
echo "Tagged xclbins:"
ls -la /home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_*_{qwen3_0_6b,qwen3_vl_4b,qwen3_8b,llama,gemma4_e2b}.xclbin 2>/dev/null | awk '{printf "  %-55s %s\n", $NF, $5}'
echo ""
echo "Total: $(ls /home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_*_{qwen3_0_6b,qwen3_vl_4b,qwen3_8b,llama,gemma4_e2b}.xclbin 2>/dev/null | wc -l) xclbins"
