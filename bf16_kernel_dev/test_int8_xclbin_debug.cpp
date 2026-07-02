// test_int8_xclbin_debug.cpp - Debug INT8 xclbin output
// Tests with a very simple matrix to see what the NPU actually computes
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

static std::vector<uint32_t> load_insts(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(sz / 4);
    fread(v.data(), 4, v.size(), f);
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const char* xclbin_path = argc > 1 ? argv[1] 
        : "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV.xclbin";
    const char* insts_path = argc > 2 ? argv[2] 
        : "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV.txt";
    
    int M = argc > 3 ? atoi(argv[3]) : 128;
    int K = argc > 4 ? atoi(argv[4]) : 1024;
    int N = argc > 5 ? atoi(argv[5]) : 1024;
    
    printf("INT8 DEBUG Test: %s\n", xclbin_path);
    printf("Matrix: %dx%dx%d\n", M, K, N);
    
    auto instr = load_insts(insts_path);
    if (instr.empty()) return 1;
    printf("Instructions: %zu words\n", instr.size());
    
    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");
    
    // Create BOs matching the fused engine's group_id convention
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;
    
    // Use EXACT same pattern as fused engine
    auto bo_instr = xrt::bo(device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(device, a_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, b_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(device, c_sz * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    
    // Copy instructions
    memcpy(bo_instr.map(), instr.data(), instr.size() * 4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size() * 4, 0);
    
    // Fill A with all 1s
    auto a_map = bo_a.map<int8_t*>();
    memset(a_map, 1, a_sz);
    
    // Fill B with all 1s
    auto b_map = bo_b.map<int8_t*>();
    memset(b_map, 1, b_sz);
    
    // Zero C
    auto c_map = bo_c.map<int16_t*>();
    memset(c_map, 0, c_sz * sizeof(int16_t));
    
    // Sync
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, a_sz, 0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, b_sz, 0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, c_sz * sizeof(int16_t), 0);
    
    // Run
    printf("Running kernel...\n");
    auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_sz * sizeof(int16_t), 0);
    
    // Expected: C[m][n] = sum over k of 1*1 = K = 1024
    // Check a few elements
    printf("\nExpected: C[m][n] = %d (all ones)\n", K);
    printf("\nFirst 20 C values:\n");
    for (int i = 0; i < std::min((int)c_sz, 20); i++) {
        printf("  C[%d] = %d\n", i, c_map[i]);
    }
    
    // Also check if C looks like a scaled version of something
    int64_t sum = 0;
    int64_t sum_sq = 0;
    int nonzero = 0;
    for (int i = 0; i < std::min((int)c_sz, 1000); i++) {
        if (c_map[i] != 0) nonzero++;
        sum += c_map[i];
        sum_sq += (int64_t)c_map[i] * c_map[i];
    }
    printf("\nFirst 1000 elements: %d non-zero, avg=%.2f, std=%.2f\n", 
           nonzero, (double)sum/nonzero, sqrt((double)(sum_sq - sum*sum/nonzero)/nonzero));
    
    // Check if there's a pattern in C
    printf("\nC[0..127] (first full row):\n");
    for (int n = 0; n < std::min(N, 16); n++) {
        printf("  %4d", c_map[n]);
    }
    printf("\n");
    
    // Check C[128..255] (second full row)
    printf("\nC[128..143] (second row):\n");
    for (int n = 0; n < std::min(N, 16); n++) {
        printf("  %4d", c_map[128 + n]);
    }
    printf("\n");
    
    return 0;
}
