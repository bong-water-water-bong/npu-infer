// BF16 GEMM kernel v2 — proper AIE API with accumulators
// Tile: A[32x64], B[64x128], C[128x128]
// Uses aie::accum<accfloat, 64> for MAC operations

#include <aie_api/aie.hpp>

#define DIM_M 128
#define DIM_K 64  
#define DIM_N 128

extern "C" {

void zero_kernel_bf16_v2(bfloat16 *__restrict pC) {
  constexpr int n = DIM_M * DIM_N;
  constexpr int r = 64;
  aie::vector<bfloat16, r> zeros = aie::zeros<bfloat16, r>();
  for (int i = 0; i < n; i += r) {
    aie::store_v(pC + i, zeros);
  }
}

static int g_counter = 0;

void matmul_bf16_v2(bfloat16 *__restrict pA,
                    bfloat16 *__restrict pB,
                    bfloat16 *__restrict pC) {
  // C row offset (same g_counter pattern as working mm_bfp_mixed.cc)
  constexpr int m = 32;
  constexpr int n = 128;
  bfloat16 *pC_slice = pC + g_counter * m * n;
  if (g_counter == 3) g_counter = 0; else g_counter++;

  // Use 8x8x8 mmul via accumulators
  constexpr int r = 8, s = 8, t = 8;
  
  for (int i = 0; i < m / r; i++) {
    for (int j = 0; j < n / t; j++) {
      // Load A sub-block [r,s] = [8,8]
      aie::vector<bfloat16, r * s> a_vec;
      for (int ri = 0; ri < r; ri++) {
        for (int si = 0; si < s; si++) {
          a_vec[ri * s + si] = pA[(i * r + ri) * DIM_K + j * s + si];
        }
      }
      
      // Load B sub-block [s,t] = [8,8]  
      aie::vector<bfloat16, s * t> b_vec;
      for (int si = 0; si < s; si++) {
        for (int ti = 0; ti < t; ti++) {
          b_vec[si * t + ti] = pB[si * DIM_N + j * t + ti];
        }
      }
      
      // Load C accumulator (explicit constructor from vector)
      aie::accum<accfloat, r * t> acc(aie::load_v<r * t>(&pC_slice[(i * r) * n + j * t]));
      
      // MAC: acc += A × B
      acc = aie::mac(acc, a_vec, b_vec);
      
      // Store back
      aie::store_v(&pC_slice[(i * r) * n + j * t], acc.template to_vector<bfloat16>());
    }
  }
}

}  // extern "C"
