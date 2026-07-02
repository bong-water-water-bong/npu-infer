# Firmware Column Limit Investigation

## Firmware Files
Located at `/home/bcloud/amdxdna-dkms/firmware/amdnpu/17f0_11/`

| File | Size | Description |
|------|------|-------------|
| `npu.sbin.1.0.0.166` | 376,768 B | Original firmware — NO column validation |
| `npu.sbin.1.1.2.65` (active) | 429,680 B | Current firmware — has column count check |
| `npu.sbin` (symlink) | → `1.1.2.65` | Active firmware link |

## Column Validation

**1.1.2.65** contains `"Invalid column count: %u >= %u"` at binary offset `0x1d6d1`. 
This validates the partition column count against a hardcoded limit (8) in the PSP firmware.
ALL attempts to create HW contexts with `column_width > 8` get `EINVAL`.

**1.0.0.166** has NO column validation strings — no `"column count"`, no `"Invalid col"`.
This suggests it may accept partitions wider than 8 columns.

## Path to >8 Columns

1. Load older firmware: `cp npu.sbin.1.0.0.166 npu.sbin`
2. Or patch the validation string in 1.1.2.65 at offset 0x1d6d1
3. Then try 12/16/40 column xclbins

Risk: older firmware may lack bug fixes, performance optimizations, or SMU init ordering.

## 31 TFLOPS is the confirmed 8-column max
Config2 with 192×128×96 tiles, 32 cores = 31.0 TFLOPS.
To go beyond (50+ TFLOPS), need >8 columns or a wider design.
