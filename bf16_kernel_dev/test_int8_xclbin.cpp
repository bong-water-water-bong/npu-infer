// test_int8_xclbin.cpp - Test INT8 xclbin on AMD NPU
// Compile: g++ -std=c++23 -O3 -o test_int8_xclbin test_int8_xclbin.cpp \
//   -I/home/bcloud/torch2aie/toolchain/xrt/include \
//   -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
//   -lxrt_coreutil -luuid -lm
// Run: LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64 ./test_int8_xclbin
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

struct XCLBIN_SPEC {
    const char* path;
    const char* name;
    int M, K, N;  // Full matrix dimensions
    int tile_m, tile_k, tile_n;  // Tile dimensions
};

static const XCLBIN_SPEC specs[] = {
    //{"/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV.xclbin",  "KV",  128, 1024, 1024, 128, 64, 128},
    //{"/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV.xclbin", "QKV", 128, 1024, 4096, 128, 64, 128},
    //{"/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_GU.xclbin",  "GU",  128, 1024, 6144, 128, 64, 128},
    {"/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_D.xclbin",   "D",   128, 3072, 1024, 128, 64, 128},
    //{"/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_O.xclbin",   "O",   128, 2048, 1024, 128, 64, 128},
};

// Reference GEMM: C += A * B  (all row-major, int8 matmul)
static void ref_gemm_int8(int8_t* A, int8_t* B, int16_t* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] += (int16_t)sum;
        }
    }
}

int main() {
    std::cout << "=== INT8 XCLBIN Test ===\n";
    
    // Connect to NPU device
    auto device = xrt::device(0);
    std::cout << "Device: " << device.get_info(xrt::info::device::name) << "\n";
    
    for (auto& spec : specs) {
        std::cout << "\n--- Testing " << spec.name << " (" << spec.path << ") ---\n";
        
        // Load xclbin
        auto uuid = device.load_xclbin(spec.path);
        std::cout << "  UUID: " << uuid.to_string() << "\n";
        
        // Get kernel
        auto kernel = xrt::kernel(device, uuid, "MLIR_AIE");
        
        // Calculate sizes
        int M = spec.M;
        int K = spec.K;
        int N = spec.N;
        int a_size = M * K;
        int b_size = K * N;
        int c_size = M * N;
        
        // Create BOs
        auto bo_a = xrt::bo(device, a_size, kernel.group_id(1));
        auto bo_b = xrt::bo(device, b_size, kernel.group_id(1));
        auto bo_c = xrt::bo(device, c_size * sizeof(int16_t), kernel.group_id(1));
        auto bo_instr = xrt::bo(device, 4096, kernel.group_id(1));
        
        // Fill input data
        auto a_mapped = bo_a.map<int8_t*>();
        auto b_mapped = bo_b.map<int8_t*>();
        auto c_mapped = bo_c.map<int16_t*>();
        
        for (int i = 0; i < a_size; i++)
            a_mapped[i] = (int8_t)((i * 7 + 13) % 63 - 31);
        for (int i = 0; i < b_size; i++)
            b_mapped[i] = (int8_t)((i * 3 + 7) % 63 - 31);
        memset(c_mapped, 0, c_size * sizeof(int16_t));
        
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, a_size, 0);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, b_size, 0);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, c_size * sizeof(int16_t), 0);
        
        // Prepare instructions from the insts file
        std::string insts_path = std::string(spec.path);
        insts_path.replace(insts_path.end() - 6, insts_path.end(), ".txt");
        insts_path.replace(insts_path.find("final_i8_"), 9, "insts_i8_");
        
        printf("  Looking for insts at: %s\n", insts_path.c_str());
        
        FILE* f = fopen(insts_path.c_str(), "r");
        if (!f) {
            printf("  ❌ Cannot open %s\n", insts_path.c_str());
            continue;
        }
        
        std::vector<uint32_t> instr;
        uint32_t val;
        while (fscanf(f, "%x,", &val) == 1) {
            instr.push_back(val);
        }
        fclose(f);
        
        printf("  Instructions: %zu words\n", instr.size());
        
        bo_instr.write(instr.data(), instr.size() * 4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size() * 4, 0);
        
        // Run kernel
        auto run = kernel(
            0,                           // opcode
            bo_instr,                    // instruction BO
            instr.size(),                // instruction count
            bo_a, bo_b, bo_c             // a, b, c BOs
        );
        
        run.wait();
        std::cout << "  Kernel completed\n";
        
        // Read back result
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_size * sizeof(int16_t), 0);
        
        // Verify
        std::vector<int16_t> ref(c_size, 0);
        ref_gemm_int8(a_mapped, b_mapped, ref.data(), M, K, N);
        
        int errors = 0;
        int64_t max_diff = 0;
        double total_diff = 0;
        int64_t max_abs_ref = 0;
        for (int i = 0; i < std::min(c_size, 10000); i++) {
            int64_t diff = std::abs((int64_t)c_mapped[i] - (int64_t)ref[i]);
            max_diff = std::max(max_diff, diff);
            total_diff += diff;
            int64_t abs_ref = std::abs((int64_t)ref[i]);
            max_abs_ref = std::max(max_abs_ref, abs_ref);
            if (std::abs(diff) > 1) errors++;
        }
        
        printf("  Verification (first %d elements):\n", std::min(c_size, 10000));
        printf("    Ref[0]: %d  NPU[0]: %d\n", ref[0], c_mapped[0]);
        printf("    Ref[1]: %d  NPU[1]: %d\n", ref[1], c_mapped[1]);
        printf("    Ref[2]: %d  NPU[2]: %d\n", ref[2], c_mapped[2]);
        printf("    Errors (>1 diff): %d / %d\n", errors, std::min(c_size, 10000));
        printf("    Max diff: %ld\n", max_diff);
        printf("    Avg diff: %.2f\n", total_diff / std::min(c_size, 10000));
        printf("    Max |ref|: %ld\n", max_abs_ref);
        
        if (errors == 0 && max_diff <= 1) {
            printf("  ✅ PASS\n");
        } else {
            printf("  ❌ FAIL (%d errors, max diff %ld)\n", errors, (long)max_diff);
        }
    }
    
    return 0;
}
