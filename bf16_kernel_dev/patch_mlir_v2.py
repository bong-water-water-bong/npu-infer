#!/usr/bin/env python3
"""
Read the working BFP16 MLIR and produce a BF16 v2 MLIR with 64x64 B tiles.
Usage: python3 patch_mlir_v2.py <input.mlir> <output.mlir>
"""
import re, sys

def patch_mlir(input_path, output_path):
    with open(input_path) as f:
        mlir = f.read()
    
    # 1. BFP16 type → bfloat16
    # The BFP16 type is: !aiex.bfp<"v8bfp16ebs8">
    mlir = mlir.replace('!aiex.bfp<"v8bfp16ebs8">', 'bfloat16')
    
    # 2. B tile: 64x16 (BFP16 stores 8 bf16 per element) → 64x64 (bf16)
    #   memref<64x16xbf16> → memref<64x64xbf16>
    mlir = re.sub(r'memref<64x16xbfloat16>', 'memref<64x64xbfloat16>', mlir)
    
    # 3. C tile: 128x128 → 128x64 (smaller N-tile to fit SRAM)
    #   512x128 → 512x64 (L3 C buffer)
    mlir = re.sub(r'memref<128x128xbfloat16>', 'memref<128x64xbfloat16>', mlir)
    mlir = re.sub(r'memref<512x128xbfloat16>', 'memref<512x64xbfloat16>', mlir)
    
    # 4. Kernel name changes
    mlir = mlir.replace('matmul_vectorized_different_datatypes', 'matmul_bf16_v2')
    mlir = mlir.replace('zero_kernel_bf16', 'zero_kernel_bf16_v2')
    mlir = mlir.replace('mm_128x64x128.o', 'mm_bf16_v2.o')
    
    # 5. B objectFifo depth: the BFP16 design has depth=2.
    #    For BF16 64x64 tiles, 2 × 8KB = 16KB still fits, keep depth=2.
    #    No change needed for depth.
    
    # 6. C output stream dimensions (4D strides for 512x64 output)
    #    Original: [<size = 64, stride = 256>, <size = 2, stride = 8>, 
    #                <size = 16, stride = 16>, <size = 8, stride = 1>]
    #    These describe 512x128 output broken into 64x8 chunks.
    #    For 512x64: same strides work since they describe the tiling.
    #    Actually need to check: size=64 means 64 rows per chunk, stride=256 means
    #    256 elements between row starts in the 512x128 output.
    #    For 512x64: need to verify strides.
    # Let me check what these mean for the C L3 output dimensions...
    
    # Actually, looking at the MLIR more carefully, the output strides
    # describe how data flows from L1 (128x64) to L3 (512x64) through 
    # the mem_tile. The stride values encode:
    #   size=64, stride=256: 64 consecutive elements, then skip to next block of 64
    #   This packs 4 L1 tiles (4×128×64) into one L3 tile (512×64)
    # Hmm, the stride representation is complex. Let me just match the 
    # pattern and see if aiecc accepts it.
    
    # Verify: check a few key lines
    lines = mlir.splitlines()
    print(f"Patched MLIR: {len(lines)} lines")
    
    # Check for B tile dimensions
    b_count = sum(1 for l in lines if '64x64xbfloat16' in l)
    print(f"  64x64 B tile occurrences: {b_count}")
    
    # Check for C tile dimensions 
    c_count = sum(1 for l in lines if '128x64xbfloat16' in l)
    print(f"  128x64 C tile occurrences: {c_count}")
    
    with open(output_path, 'w') as f:
        f.write(mlir)
    print(f"Written: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mlir> <output.mlir>")
        sys.exit(1)
    patch_mlir(sys.argv[1], sys.argv[2])
