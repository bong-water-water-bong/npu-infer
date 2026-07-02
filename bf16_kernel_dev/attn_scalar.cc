// Minimal NPU attention kernel — scalar, Peano-compilable.
// Q[NH*HD bf16] + KV[NKV*C*HD*2 bf16] → Out[NH*HD bf16]
// KV layout: first NKV*C*HD bf16 = K, next NKV*C*HD bf16 = V
// NH=16, NKV=8, HD=128, GQA=2, C = context length
#include <stdint.h>

// Fast exp approximation
static inline float fast_exp(float value) {
    if (value <= -20.0f) return 0.0f;
    if (value >= 0.0f) return 1.0f;
    float inv_ln2 = 1.4426950409f;
    float ln2 = 0.6931471806f;
    int exponent = (int)(value * inv_ln2);
    float reduced = value - (float)exponent * ln2;
    float r2 = reduced * reduced;
    float poly = 1.0f + reduced + 0.5f * r2 + 0.16666667f * r2 * reduced + 0.04166667f * r2 * r2;
    if (exponent < -126) return 0.0f;
    if (exponent > 127) exponent = 127;
    union { uint32_t u; float f; } bits;
    bits.u = (uint32_t)(exponent + 127) << 23;
    return bits.f * poly;
}

extern "C" void attn_scalar_bf16(
    int32_t *q_buf,     // [NH*HD/2] = 1024 i32, holds 2048 bf16
    int32_t *kv_buf,    // [NKV*C*HD] i32 = K(V) side-by-side: first half=K, second half=V
    int32_t *out_buf,   // [NH*HD/2] i32
    int32_t C            // KV context length
) {
    static const int NH=16, NKV=8, HD=128, GQA=NH/NKV;
    static const int K_DW = NKV * 128 * HD / 2; // max K size in i32 for C=128

    uint16_t *q = (uint16_t*)q_buf;
    uint16_t *k = (uint16_t*)kv_buf;               // first half
    uint16_t *v = (uint16_t*)(kv_buf + K_DW);      // second half
    uint16_t *o = (uint16_t*)out_buf;

    for (int h = 0; h < NH; h++) {
        int kvh = h / GQA;

        // QK^T scores
        float scores[128];
        float max_s = -1e30f;
        for (int pos = 0; pos < C; pos++) {
            float s = 0.0f;
            int qoff=h*HD, koff=kvh*C*HD+pos*HD;
            for (int d = 0; d < HD; d++) {
                uint32_t qb=((uint32_t)q[qoff+d])<<16; float qf; __builtin_memcpy(&qf,&qb,4);
                uint32_t kb=((uint32_t)k[koff+d])<<16; float kf; __builtin_memcpy(&kf,&kb,4);
                s += qf * kf;
            }
            s *= 0.0883883476f;
            scores[pos] = s;
            if (s > max_s) max_s = s;
        }

        // Softmax
        float sum_w = 0.0f, weights[128];
        for (int pos = 0; pos < C; pos++) {
            float d = scores[pos] - max_s;
            float w = (d > -20.0f) ? fast_exp(d) : 0.0f;
            weights[pos] = w;
            sum_w += w;
        }
        float inv_sum = sum_w > 0.0f ? 1.0f / sum_w : 1.0f / (float)C;

        // V-weight → output
        for (int d = 0; d < HD; d++) {
            float acc = 0.0f;
            int voff=kvh*C*HD; int ooff=h*HD+d;
            for (int pos = 0; pos < C; pos++) {
                uint32_t vb=((uint32_t)v[voff+pos*HD+d])<<16; float vf; __builtin_memcpy(&vf,&vb,4);
                acc += weights[pos] * inv_sum * vf;
            }
            uint32_t ob; __builtin_memcpy(&ob,&acc,4);
            o[ooff] = (uint16_t)(ob >> 16);
        }
    }
}
