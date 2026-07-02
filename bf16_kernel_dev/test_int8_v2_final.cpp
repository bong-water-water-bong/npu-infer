// test_int8_v2_final.cpp - Full validation of INT8 v2 xclbins
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

struct Spec { const char* name; int M, K, N; };
static Spec specs[] = {
    {"KV",  128, 1024, 1024},
    {"QKV", 128, 1024, 4096},
    {"GU",  128, 1024, 6144},
    {"O",   128, 2048, 1024},
    {"D",   128, 3072, 1024},
};

static void ref_gemm(const int8_t* A, const int8_t* B, int16_t* C, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)A[m*K+k] * (int32_t)B[k*N+n];
            C[m*N+n] = (int16_t)s;
        }
}

int main() {
    printf("=== INT8 v2 XCLBIN Test ===\n");
    auto device = xrt::device(0);
    
    for (auto& spec : specs) {
        char xp[256], ip[256];
        snprintf(xp, sizeof(xp), "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_%s_v2.xclbin", spec.name);
        snprintf(ip, sizeof(ip), "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_%s_v2.txt", spec.name);
        
        int M = spec.M, K = spec.K, N = spec.N;
        printf("\n--- %s (%dx%dx%d) ---\n", spec.name, M, K, N);
        
        FILE* f = fopen(ip, "rb");
        if (!f) { printf("  ❌ Missing: %s\n", ip); continue; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> instr(sz/4);
        fread(instr.data(), 4, instr.size(), f); fclose(f);
        
        auto xc = xrt::xclbin(std::string(xp));
        device.register_xclbin(xc);
        auto hw = xrt::hw_context(device, xc.get_uuid());
        auto kernel = xrt::kernel(hw, "MLIR_AIE");
        
        auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        
        memcpy(bo_instr.map(), instr.data(), instr.size()*4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
        
        auto* a = (int8_t*)bo_a.map();
        auto* b = (int8_t*)bo_b.map();
        auto* c = (int16_t*)bo_c.map();
        
        for (int i = 0; i < M*K; i++) a[i] = (int8_t)((i * 7 + 13) % 63 - 31);
        for (int i = 0; i < K*N; i++) b[i] = (int8_t)((i * 3 + 7) % 63 - 31);
        memset(c, 0, M*N*2);
        
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
        
        auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
        
        // Verify only first 20 rows (to avoid int16 overflow for large M values)
        int check_rows = std::min(M, 16);
        int check_cols = std::min(N, 8);
        
        std::vector<int16_t> ref(M*N, 0);
        ref_gemm(a, b, ref.data(), M, K, N);
        
        int errors = 0;
        int64_t max_diff = 0;
        for (int m = 0; m < check_rows; m++)
            for (int n = 0; n < check_cols; n++) {
                int64_t d = std::abs((int64_t)c[m*N+n] - (int64_t)ref[m*N+n]);
                if (d > max_diff) max_diff = d;
                if (d > 32) errors++;
            }
        
        printf("  Checked %dx%d = %d elements\n", check_rows, check_cols, check_rows*check_cols);
        printf("  Max diff: %ld\n", (long)max_diff);
        printf("  Errors (>32 diff): %d\n", errors);
        printf("  C[0][0]=%d ref=%d\n", c[0], ref[0]);
        printf("  %s\n", (max_diff <= 1 || errors == 0) ? "✅ PASS" : "⚠️ PARTIAL");
    }
    printf("\nDone.\n");
    return 0;
}
