// test_int8_xclbin_simple.cpp - Quick INT8 xclbin test
// Compile:
// g++ -std=c++23 -O3 -o test_int8_xclbin_simple test_int8_xclbin_simple.cpp \
//   -I/home/bcloud/torch2aie/toolchain/xrt/include \
//   -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
//   -lxrt_coreutil -luuid -lm
// Run:
// LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64 ./test_int8_xclbin_simple
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

// INT8 reference GEMM
static void ref_gemm(const int8_t* A, const int8_t* B, int16_t* C, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            C[m * N + n] += (int16_t)sum;
        }
}

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
    
    int M = 128, K = 1024, N = 1024;
    if (argc > 2) M = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) N = atoi(argv[4]);
    
    printf("INT8 XCLBIN Test: %s\n", xclbin_path);
    printf("Matrix: %dx%dx%d\n", M, K, N);
    
    // Load instructions
    std::string insts_path(xclbin_path);
    auto p = insts_path.rfind(".xclbin");
    if (p != std::string::npos) {
        insts_path.replace(p, 7, ".txt");
        // Fix the path format: final_i8_*.txt vs insts_i8_*.txt
        auto q = insts_path.rfind("final_i8_");
        if (q != std::string::npos)
            insts_path.replace(q, 9, "insts_i8_");
    }
    printf("Insts: %s\n", insts_path.c_str());
    
    auto instr = load_insts(insts_path.c_str());
    if (instr.empty()) { fprintf(stderr, "Failed to load instructions\n"); return 1; }
    printf("Instructions: %zu words\n", instr.size());
    
    // Connect device
    auto device = xrt::device(0);
    printf("Device: NPU2\n");
    
    // Load xclbin using same pattern as fused engine
    auto xc = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");
    
    // Allocate buffers
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;
    
    auto bo_instr = xrt::bo(device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(device, a_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, b_sz * 9 / 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4)); // BFP16 packed size
    auto bo_c = xrt::bo(device, c_sz * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    
    // Copy instructions
    memcpy(bo_instr.map(), instr.data(), instr.size() * 4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size() * 4, 0);
    
    // Fill input A
    auto a_map = bo_a.map<int8_t*>();
    for (size_t i = 0; i < a_sz; i++)
        a_map[i] = (int8_t)((i * 7 + 13) % 63 - 31);
    
    // Fill input B (raw int8, NOT bfp16 packed)
    auto b_map = bo_b.map<int8_t*>();
    for (size_t i = 0; i < b_sz; i++)
        b_map[i] = (int8_t)((i * 3 + 7) % 63 - 31);
    
    // Zero output C
    auto c_map = bo_c.map<int16_t*>();
    memset(c_map, 0, c_sz * sizeof(int16_t));
    
    // Compute reference
    std::vector<int16_t> ref(c_sz, 0);
    ref_gemm(a_map, b_map, ref.data(), M, K, N);
    
    // Sync BOs to device
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, a_sz, 0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, b_sz, 0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, c_sz * sizeof(int16_t), 0);
    
    // Run kernel (opcode=3)
    printf("Running kernel...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("Kernel: %.3f ms\n", dt);
    
    // Read back
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_sz * sizeof(int16_t), 0);
    
    // Verify
    int errors = 0;
    int64_t max_diff = 0;
    double total_diff = 0;
    int display_count = std::min((int)c_sz, 10);
    
    printf("\nFirst %d results:\n", display_count);
    for (int i = 0; i < display_count; i++) {
        printf("  [%d] ref=%5d, npu=%5d, diff=%ld\n", i, ref[i], c_map[i], 
               (long)std::abs((int64_t)c_map[i] - (int64_t)ref[i]));
    }
    
    int check_count = std::min((int)c_sz, 5000);
    for (int i = 0; i < check_count; i++) {
        int64_t diff = std::abs((int64_t)c_map[i] - (int64_t)ref[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
        if (diff > 1) errors++;
    }
    
    printf("\nVerification (first %d of %d elements):\n", check_count, (int)c_sz);
    printf("  Errors (>1 diff): %d / %d\n", errors, check_count);
    printf("  Max diff: %ld\n", (long)max_diff);
    printf("  Avg diff: %.3f\n", total_diff / check_count);
    
    if (errors == 0 && max_diff <= 1)
        printf("\n✅ PASS - INT8 xclbin works correctly!\n");
    else
        printf("\n❌ FAIL - %d errors, max diff %ld\n", errors, (long)max_diff);
    
    return errors == 0 ? 0 : 1;
}
