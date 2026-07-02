// SiLU gate kernel for Qwen3-0.6B: gate(3072) + up(3072) → sigmoid(gate)*up → down_input(3072)
// All bf16, Peano-compilable, minimal
#include <stdint.h>

extern "C" void silu_gate_bf16(
    int32_t *gu_out,   // [6144 bf16] = 3072 i32: first 1536 bf16=gate, next 1536 bf16=up
    int32_t *d_in,     // [3072 bf16] = 1536 i32: silu(gate)*up output
    int32_t n          // number of elements = 3072
) {
    uint16_t *gu = (uint16_t*)gu_out;
    uint16_t *d = (uint16_t*)d_in;

    for (int i = 0; i < n; i++) {
        // gate value at gu[i], up value at gu[i + n]
        uint32_t gb = ((uint32_t)gu[i]) << 16;
        float gate; __builtin_memcpy(&gate, &gb, 4);

        uint32_t ub = ((uint32_t)gu[i + n]) << 16;
        float up; __builtin_memcpy(&up, &ub, 4);

        // sigmoid approx: 1/(1+exp(-x)) using fast exp
        float sig;
        if (gate <= -10.0f) sig = 0.0f;
        else if (gate >= 10.0f) sig = 1.0f;
        else {
            // fast_exp(-|x|) then sigmoid
            float ax = gate < 0 ? -gate : gate;
            float inv_ln2 = 1.4426950409f, ln2 = 0.6931471806f;
            int e = (int)(ax * inv_ln2);
            float r = ax - (float)e * ln2;
            float r2 = r*r;
            float poly = 1.0f + r + 0.5f*r2 + 0.16666667f*r2*r + 0.04166667f*r2*r2;
            if (e > 127) e = 127;
            union { uint32_t u; float f; } b;
            b.u = (uint32_t)(-e + 127) << 23;
            float exp_neg = b.f * poly;

            if (gate >= 0) sig = 1.0f / (1.0f + exp_neg);
            else sig = exp_neg / (1.0f + exp_neg);
        }

        float result = sig * up;
        uint32_t ob; __builtin_memcpy(&ob, &result, 4);
        d[i] = (uint16_t)(ob >> 16);
    }
}
