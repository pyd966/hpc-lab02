// Main task: optimize the MoE forward pass.

#include "moe.h"

#include <cmath>
#include <cstddef>

static bool has_shape(const MoEWeights& w, int d_model, int d_ff,
                      int num_experts, int top_k) {
    return w.d_model == d_model && w.d_ff == d_ff &&
           w.num_experts == num_experts && w.top_k == top_k;
}

static void preprocess_s1(MoEWeights& w) {}

static void preprocess_s2(MoEWeights& w) {}

static void preprocess_s3(MoEWeights& w) {}

static void preprocess_s4(MoEWeights& w) {}

void preprocess(MoEWeights& w) {
    if (has_shape(w, 256, 128, 16, 4)) {
        // S1 and S3 differ only in num_tokens, which is not available here.
        preprocess_s1(w);
        preprocess_s3(w);
        return;
    }
    if (has_shape(w, 1024, 512, 16, 4)) {
        preprocess_s2(w);
        return;
    }
    if (has_shape(w, 512, 128, 512, 2)) {
        preprocess_s4(w);
    }
}

static void my_expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {
    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;
    for (int f = 0; f < d_ff; f++) {
        int32_t acc_g = 0;
        int32_t acc_u = 0;
        for (int d = 0; d < d_model; d++) {
            acc_g += (int32_t)w_gate[f * d_model + d] * (int32_t)xq[d];
            acc_u += (int32_t)w_up[f * d_model + d] * (int32_t)xq[d];
        }
        float vg = (float)acc_g * (s_x * s_gate);
        float vu = (float)acc_u * (s_x * s_up);
        float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f++) {
        hq[f] = (int8_t)lrintf(h[f] / s_h);
    }

    // Down projection
    for (int d = 0; d < d_model; d++) {
        int32_t acc = 0;
        for (int f = 0; f < d_ff; f++) {
            acc += (int32_t)w_down[d * d_ff + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
    }
}

static void moe_forward_generic(const float* x, const MoEWeights& w, float* y,
                                int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    #pragma omp parallel for
    for (int t = 0; t < num_tokens; t++) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;

        // 1. Affinity scores
        float s[MAX_NUM_EXPERTS];
        for (int e = 0; e < num_experts; e++) {
            float acc = 0.0f;
            for (int d = 0; d < d_model; d++) {
                acc += w.w_router[(size_t)e * d_model + d] * xt[d];
            }
            s[e] = 1.0f / (1.0f + expf(-acc));
        }

        // 2. Top-K selection by biased score (ties broken by smaller index)
        int topk_idx[MAX_TOP_K];
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; k++) {
            int best = -1;
            for (int e = 0; e < num_experts; e++) {
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
        }

        // 3. Gate values: normalize the ORIGINAL affinities of the selected
        //    experts (the bias never enters the gate values)
        float gate_sum = 0.0f;
        for (int k = 0; k < top_k; k++) gate_sum += s[topk_idx[k]];

        // 4. Quantize the token to int8 (symmetric, per-token scale)
        float x_amax = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float a = fabsf(xt[d]);
            if (a > x_amax) x_amax = a;
        }
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; d++) {
            xq[d] = (int8_t)lrintf(xt[d] / s_x);
        }

        // 5+6. Shared expert (always on), then selected routed experts,
        //      combined on top of the residual connection
        float o[MAX_D_MODEL];
        my_expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                   w.sh_s_down, xq, s_x, o, d_model, d_ff);
        for (int d = 0; d < d_model; d++) {
            yt[d] = xt[d] + o[d];
        }

        for (int k = 0; k < top_k; k++) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            my_expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                       w.w_up + (size_t)e * d_ff * d_model,
                       w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                       w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; d++) {
                yt[d] += gate * o[d];
            }
        }
    }
}

static void moe_forward_optimized_s1(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
    moe_forward_generic(x, w, y, num_tokens);
}

static void moe_forward_optimized_s2(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
    moe_forward_generic(x, w, y, num_tokens);
}

static void moe_forward_optimized_s3(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
    moe_forward_generic(x, w, y, num_tokens);
}

static void moe_forward_optimized_s4(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
    moe_forward_generic(x, w, y, num_tokens);
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (num_tokens == 1 && has_shape(w, 256, 128, 16, 4)) {
        moe_forward_optimized_s1(x, w, y, num_tokens);
        return;
    }
    if (num_tokens == 1 && has_shape(w, 1024, 512, 16, 4)) {
        moe_forward_optimized_s2(x, w, y, num_tokens);
        return;
    }
    if (num_tokens == 128 && has_shape(w, 256, 128, 16, 4)) {
        moe_forward_optimized_s3(x, w, y, num_tokens);
        return;
    }
    if (num_tokens == 1024 && has_shape(w, 512, 128, 512, 2)) {
        moe_forward_optimized_s4(x, w, y, num_tokens);
        return;
    }

    moe_forward_generic(x, w, y, num_tokens);
}
