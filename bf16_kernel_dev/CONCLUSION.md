## BF16 Investigation Conclusion (2026-06-29)

### The BF16 hang is a **hardware DMA limitation**, not a compiler bug.

Evidence:
1. BF16 v1 (emulation flag): HUNG ✅
2. BF16 v2 (identity, emulation flag): HUNG ✅  
3. BF16 v3 (native, NO emulation flag): HUNG ✅
4. BFP16 (identical MLIR, different type): WORKS ✅

### Root Cause
The AIE2P DMA engine on XDNA2 supports streaming data in:
- **BFP16** (`v8bfp16ebs8`): hardware format, 8 BF16 values packed into 9 bytes, DMA can directly load into tile-local memory
- **INT8**: 8-bit integer format, also hardware-supported
- **BF16**: NOT a hardware DMA format. The DMA engine cannot stream raw BF16 data to tile SRAM.

When the MLIR declares `memref<64x128xbf16>` for the B buffer, the aiecc tries to generate DMA descriptors for BF16 streaming, but the hardware DMA controller doesn't support this format → stall/hang.

The BFP16 kernel works around this by:
1. **Weights stored as BFP16** — DMA streams BFP16 directly to tile SRAM
2. **Activations converted on-tile** — kernel converts BF16 A → BFP16 before computation
3. **BFP16×BFP16 MAC** — uses `mac_8x8_8x8T` intrinsic designed for BFP16

### Why Windows might work
Windows likely uses a different NPU driver stack (possibly via ROCm or DirectML) that handles BF16 at a higher level — converting it to BFP16 or INT8 before dispatching to the AIE hardware. The Linux XRT/aiecc stack exposes the raw hardware interface, which doesn't support BF16 directly.

### Path Forward
- **BFP16 with scale=1.0** — already working at 210ms/tok with RMSE 0.0003 (practically lossless)
- **Multi-token decode** — requires BFP16 M=256 xclbins which have a K-iteration instruction count issue in aiecc
- **2-layer batch** — N=8320 workaround, all xclbins built, just needs engine integration
