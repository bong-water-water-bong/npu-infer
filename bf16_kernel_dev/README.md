# BF16 Native Kernel Development

**Status: Blocked — aiecc DMA descriptor bug for BF16 types**

## Findings (2026-06-29)

### What was tested
1. **Identity kernel** — replaced matmul with A→C copy. **Hangs.**
2. **Correct data format** — raw BF16 packing, correct BO sizes. **Hangs.**
3. **64×64 B tiles, depth=2** — redesigned MLIR with smaller B tiles. **Hangs.**

### Root Cause: aiecc BF16 type bug
Every BF16 MLIR variant hangs, regardless of tile size, depth, or kernel code.
The working BFP16 MLIR (identical structure, different types) works at 31 TFLOPS.
**The aiecc version we have generates incorrect DMA descriptors for BF16 memory types,
causing the NPU DMA controller to access out-of-bounds memory → stall/hang.**

Evidence:
- All 3 BF16 xclbins hang at the first kernel call (no output)
- BFP16 xclbin with same structure works immediately
- Even identity kernel (no computation) hangs — confirms DMA, not compute issue
- 64×64 B tiles with depth=2 still hang — proves not SRAM or back-pressure related

### Working Solution
Use BFP16 format with scale=1.0 (no ×16 hack). Precision drift is ~RMSE 0.0003.
BFP16 is the only proven format for this aiecc version.

### Files

| File | Purpose |
|------|---------|
| `mm_bf16_v2.cc` | Redesigned kernel (64×64 B tiles, proper AIE API) — compiles but hangs |
| `mm_bf16_test.cc` | Identity test kernel — proves DMA issue |
| `FIRMWARE-NOTES.md` | 1.0.0.166 firmware may accept >8 columns |
| `patch_mlir_v2.py` | MLIR patching script |
| `build_v2_direct.sh` | Full build: kernel → MLIR → aiecc → xclbin |
| `build/design_bf16_v2.xclbin` | Built BF16 v2 xclbin (150KB) — hangs due to aiecc bug |
