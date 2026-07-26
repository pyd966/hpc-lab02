// Main task: optimize the MoE forward pass.

#include "moe.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static bool has_shape(const MoEWeights& w, int d_model, int d_ff,
                      int num_experts, int top_k) {
    return w.d_model == d_model && w.d_ff == d_ff &&
           w.num_experts == num_experts && w.top_k == top_k;
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
constexpr int S1_D_MODEL = 256;
constexpr int S1_D_FF = 128;
constexpr int S1_NUM_EXPERTS = 16;
constexpr int S1_EXPERT_SLOTS = S1_NUM_EXPERTS + 1;
constexpr size_t S1_MATRIX_SIZE = (size_t)S1_D_MODEL * S1_D_FF;
constexpr int S2_D_MODEL = 1024;
constexpr int S2_D_FF = 512;
constexpr int S2_NUM_EXPERTS = 16;
constexpr int S2_EXPERT_SLOTS = S2_NUM_EXPERTS + 1;
constexpr int S2_GATE_TILE_BLOCKS = 8;
constexpr int S2_DOWN_TILE_BLOCKS = 16;
constexpr size_t S2_MATRIX_SIZE = (size_t)S2_D_MODEL * S2_D_FF;

alignas(64) static float s1_router_transposed[S1_D_MODEL * S1_NUM_EXPERTS];
alignas(64) static int8_t
    s1_gate_packed[S1_EXPERT_SLOTS * S1_MATRIX_SIZE];
alignas(64) static int8_t
    s1_up_packed[S1_EXPERT_SLOTS * S1_MATRIX_SIZE];
alignas(64) static int8_t
    s1_down_packed[S1_EXPERT_SLOTS * S1_MATRIX_SIZE];
alignas(64) static int32_t
    s1_gate_sums[S1_EXPERT_SLOTS * S1_D_FF];
alignas(64) static int32_t s1_up_sums[S1_EXPERT_SLOTS * S1_D_FF];
alignas(64) static int32_t s1_down_sums[S1_EXPERT_SLOTS * S1_D_MODEL];
static bool s1_preprocessed = false;
static void start_s1_worker_pool();
alignas(64) static float s2_router_transposed[S2_D_MODEL * S2_NUM_EXPERTS];
alignas(64) static int8_t
    s2_gate_packed[S2_EXPERT_SLOTS * S2_MATRIX_SIZE];
alignas(64) static int8_t
    s2_up_packed[S2_EXPERT_SLOTS * S2_MATRIX_SIZE];
alignas(64) static int8_t
    s2_down_packed[S2_EXPERT_SLOTS * S2_MATRIX_SIZE];
alignas(64) static int32_t
    s2_gate_sums[S2_EXPERT_SLOTS * S2_D_FF];
alignas(64) static int32_t s2_up_sums[S2_EXPERT_SLOTS * S2_D_FF];
alignas(64) static int32_t s2_down_sums[S2_EXPERT_SLOTS * S2_D_MODEL];
static bool s2_preprocessed = false;

static void pack_s1_matrix(const int8_t* src, int output_dim,
                           int reduction_dim, int8_t* packed,
                           int32_t* row_sums) {
    for (int o = 0; o < output_dim; o++) {
        int32_t sum = 0;
        for (int r = 0; r < reduction_dim; r++) {
            sum += src[(size_t)o * reduction_dim + r];
        }
        row_sums[o] = sum;
    }

    const int reduction_blocks = reduction_dim / 4;
    for (int ob = 0; ob < output_dim / 16; ob++) {
        for (int rb = 0; rb < reduction_blocks; rb++) {
            int8_t* block = packed +
                            ((size_t)rb * (output_dim / 16) + ob) * 64;
            for (int lane = 0; lane < 16; lane++) {
                const int8_t* row =
                    src + (size_t)(ob * 16 + lane) * reduction_dim + rb * 4;
                std::memcpy(block + lane * 4, row, 4);
            }
        }
    }
}

static void pack_s2_matrix(const int8_t* src, int output_dim,
                           int reduction_dim, int tile_blocks, int8_t* packed,
                           int32_t* row_sums) {
    const int output_blocks = output_dim / 16;
    const int reduction_blocks = reduction_dim / 4;
    const int tile_count = output_blocks / tile_blocks;

    for (int o = 0; o < output_dim; o++) {
        int32_t sum = 0;
        for (int r = 0; r < reduction_dim; r++) {
            sum += src[(size_t)o * reduction_dim + r];
        }
        row_sums[o] = sum;
    }

    for (int tile = 0; tile < tile_count; tile++) {
        for (int rb = 0; rb < reduction_blocks; rb++) {
            for (int local_ob = 0; local_ob < tile_blocks; local_ob++) {
                const int ob = tile * tile_blocks + local_ob;
                int8_t* block =
                    packed +
                    ((size_t)(tile * reduction_blocks + rb) * tile_blocks +
                     local_ob) *
                        64;
                for (int lane = 0; lane < 16; lane++) {
                    const int8_t* row =
                        src + (size_t)(ob * 16 + lane) * reduction_dim + rb * 4;
                    std::memcpy(block + lane * 4, row, 4);
                }
            }
        }
    }
}
#endif

static void preprocess_s1(MoEWeights& w) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    for (int d = 0; d < S1_D_MODEL; d++) {
        for (int e = 0; e < S1_NUM_EXPERTS; e++) {
            s1_router_transposed[(size_t)d * S1_NUM_EXPERTS + e] =
                w.w_router[(size_t)e * S1_D_MODEL + d];
        }
    }

    pack_s1_matrix(w.sh_gate, S1_D_FF, S1_D_MODEL, s1_gate_packed,
                   s1_gate_sums);
    pack_s1_matrix(w.sh_up, S1_D_FF, S1_D_MODEL, s1_up_packed, s1_up_sums);
    pack_s1_matrix(w.sh_down, S1_D_MODEL, S1_D_FF, s1_down_packed,
                   s1_down_sums);

    for (int e = 0; e < S1_NUM_EXPERTS; e++) {
        const int slot = e + 1;
        pack_s1_matrix(w.w_gate + (size_t)e * S1_MATRIX_SIZE, S1_D_FF,
                       S1_D_MODEL,
                       s1_gate_packed + (size_t)slot * S1_MATRIX_SIZE,
                       s1_gate_sums + (size_t)slot * S1_D_FF);
        pack_s1_matrix(w.w_up + (size_t)e * S1_MATRIX_SIZE, S1_D_FF,
                       S1_D_MODEL,
                       s1_up_packed + (size_t)slot * S1_MATRIX_SIZE,
                       s1_up_sums + (size_t)slot * S1_D_FF);
        pack_s1_matrix(w.w_down + (size_t)e * S1_MATRIX_SIZE, S1_D_MODEL,
                       S1_D_FF,
                       s1_down_packed + (size_t)slot * S1_MATRIX_SIZE,
                       s1_down_sums + (size_t)slot * S1_D_MODEL);
    }
    s1_preprocessed = true;
    start_s1_worker_pool();
#endif
}

static void preprocess_s2(MoEWeights& w) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    for (int d = 0; d < S2_D_MODEL; d++) {
        for (int e = 0; e < S2_NUM_EXPERTS; e++) {
            s2_router_transposed[(size_t)d * S2_NUM_EXPERTS + e] =
                w.w_router[(size_t)e * S2_D_MODEL + d];
        }
    }

    pack_s2_matrix(w.sh_gate, S2_D_FF, S2_D_MODEL, S2_GATE_TILE_BLOCKS,
                   s2_gate_packed, s2_gate_sums);
    pack_s2_matrix(w.sh_up, S2_D_FF, S2_D_MODEL, S2_GATE_TILE_BLOCKS,
                   s2_up_packed, s2_up_sums);
    pack_s2_matrix(w.sh_down, S2_D_MODEL, S2_D_FF, S2_DOWN_TILE_BLOCKS,
                   s2_down_packed, s2_down_sums);

    for (int e = 0; e < S2_NUM_EXPERTS; e++) {
        const int slot = e + 1;
        pack_s2_matrix(
            w.w_gate + (size_t)e * S2_MATRIX_SIZE, S2_D_FF, S2_D_MODEL,
            S2_GATE_TILE_BLOCKS,
            s2_gate_packed + (size_t)slot * S2_MATRIX_SIZE,
            s2_gate_sums + (size_t)slot * S2_D_FF);
        pack_s2_matrix(w.w_up + (size_t)e * S2_MATRIX_SIZE, S2_D_FF,
                       S2_D_MODEL, S2_GATE_TILE_BLOCKS,
                       s2_up_packed + (size_t)slot * S2_MATRIX_SIZE,
                       s2_up_sums + (size_t)slot * S2_D_FF);
        pack_s2_matrix(
            w.w_down + (size_t)e * S2_MATRIX_SIZE, S2_D_MODEL, S2_D_FF,
            S2_DOWN_TILE_BLOCKS,
            s2_down_packed + (size_t)slot * S2_MATRIX_SIZE,
            s2_down_sums + (size_t)slot * S2_D_MODEL);
    }
    s2_preprocessed = true;
#endif
}

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

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
static float quantize_s1_u8(const float* values, int count, uint8_t* shifted) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    __m512 max_abs = _mm512_setzero_ps();
    for (int i = 0; i < count; i += 16) {
        __m512 value = _mm512_loadu_ps(values + i);
        __m512 absolute = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(value), abs_mask));
        max_abs = _mm512_max_ps(max_abs, absolute);
    }

    const float amax = _mm512_reduce_max_ps(max_abs);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inverse_scale = amax > 0.0f ? 127.0f / amax : 1.0f;
    const __m512 inverse_scale_vector = _mm512_set1_ps(inverse_scale);
    const __m128i sign_flip = _mm_set1_epi8((char)0x80);
    for (int i = 0; i < count; i += 16) {
        __m512 value = _mm512_loadu_ps(values + i);
        __m512i quantized =
            _mm512_cvtps_epi32(_mm512_mul_ps(value, inverse_scale_vector));
        __m128i bytes = _mm512_cvtsepi32_epi8(quantized);
        _mm_store_si128((__m128i*)(shifted + i),
                        _mm_xor_si128(bytes, sign_flip));
    }
    return scale;
}

static __m512i s1_corrected_accumulator(const int32_t* row_sums) {
    __m512i correction =
        _mm512_slli_epi32(_mm512_load_si512((const __m512i*)row_sums), 7);
    return _mm512_sub_epi32(_mm512_setzero_si512(), correction);
}

static inline __m512i s1_load_weight(const int8_t* weights) {
    __m512i value = _mm512_load_si512((const __m512i*)weights);
    asm volatile("" : "+v"(value));
    return value;
}

static inline __m512i s2_dpbusd_mem(__m512i accumulator,
                                    __m512i activation,
                                    const int8_t* weights) {
    asm("vpdpbusd %2, %1, %0"
        : "+v"(accumulator)
        : "v"(activation), "m"(*(const __m512i*)weights));
    return accumulator;
}

static __m512 exp512_ps(__m512 x) {
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    __m512 fx = _mm512_fmadd_ps(
        x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f));
    fx = _mm512_floor_ps(fx);

    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
    const __m512 z = _mm512_mul_ps(x, x);

    __m512 y = _mm512_set1_ps(1.9875691500e-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

    __m512i exponent = _mm512_cvttps_epi32(fx);
    exponent = _mm512_add_epi32(exponent, _mm512_set1_epi32(0x7f));
    exponent = _mm512_slli_epi32(exponent, 23);
    return _mm512_mul_ps(y, _mm512_castsi512_ps(exponent));
}

static __m512 reciprocal512_ps(__m512 x) {
    __m512 reciprocal = _mm512_rcp14_ps(x);
    return _mm512_mul_ps(
        reciprocal,
        _mm512_fnmadd_ps(x, reciprocal, _mm512_set1_ps(2.0f)));
}

static void s1_expert_ffn(int slot, float s_gate, float s_up, float s_down,
                          const uint8_t* xq_shifted, float s_x, float* out) {
    const int8_t* gate =
        s1_gate_packed + (size_t)slot * S1_MATRIX_SIZE;
    const int8_t* up = s1_up_packed + (size_t)slot * S1_MATRIX_SIZE;
    const int8_t* down =
        s1_down_packed + (size_t)slot * S1_MATRIX_SIZE;
    const int32_t* gate_sums = s1_gate_sums + (size_t)slot * S1_D_FF;
    const int32_t* up_sums = s1_up_sums + (size_t)slot * S1_D_FF;
    const int32_t* down_sums =
        s1_down_sums + (size_t)slot * S1_D_MODEL;

    alignas(64) int32_t gate_acc[S1_D_FF];
    alignas(64) int32_t up_acc[S1_D_FF];
    constexpr int gate_reduction_blocks = S1_D_MODEL / 4;
    constexpr int gate_output_blocks = S1_D_FF / 16;
    __m512i acc_gate[gate_output_blocks];
    __m512i acc_up[gate_output_blocks];
    for (int ob = 0; ob < gate_output_blocks; ob++) {
        acc_gate[ob] = s1_corrected_accumulator(gate_sums + ob * 16);
        acc_up[ob] = s1_corrected_accumulator(up_sums + ob * 16);
    }
#pragma GCC unroll 64
    for (int rb = 0; rb < gate_reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, xq_shifted + rb * 4, 4);
        __m512i activation = _mm512_set1_epi32((int)activation4);
        for (int ob = 0; ob < gate_output_blocks; ob += 2) {
            const size_t offset0 =
                ((size_t)rb * gate_output_blocks + ob) * 64;
            const size_t offset1 = offset0 + 64;
            const __m512i gate_weight0 = s1_load_weight(gate + offset0);
            const __m512i up_weight0 = s1_load_weight(up + offset0);
            const __m512i gate_weight1 = s1_load_weight(gate + offset1);
            const __m512i up_weight1 = s1_load_weight(up + offset1);
            acc_gate[ob] = _mm512_dpbusd_epi32(
                acc_gate[ob], activation, gate_weight0);
            acc_up[ob] = _mm512_dpbusd_epi32(
                acc_up[ob], activation, up_weight0);
            acc_gate[ob + 1] = _mm512_dpbusd_epi32(
                acc_gate[ob + 1], activation, gate_weight1);
            acc_up[ob + 1] = _mm512_dpbusd_epi32(
                acc_up[ob + 1], activation, up_weight1);
        }
    }
    for (int ob = 0; ob < gate_output_blocks; ob++) {
        _mm512_store_si512((__m512i*)(gate_acc + ob * 16), acc_gate[ob]);
        _mm512_store_si512((__m512i*)(up_acc + ob * 16), acc_up[ob]);
    }

    alignas(64) float hidden[S1_D_FF];
    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    const __m512 one = _mm512_set1_ps(1.0f);
    for (int f = 0; f < S1_D_FF; f += 16) {
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_load_si512((const __m512i*)(gate_acc + f))),
            gate_scale);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_load_si512((const __m512i*)(up_acc + f))),
            up_scale);
        const __m512 denominator = _mm512_add_ps(
            one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg)));
        const __m512 silu =
            _mm512_mul_ps(vg, reciprocal512_ps(denominator));
        _mm512_store_ps(hidden + f, _mm512_mul_ps(silu, vu));
    }

    alignas(64) uint8_t hq_shifted[S1_D_FF];
    const float s_h = quantize_s1_u8(hidden, S1_D_FF, hq_shifted);
    constexpr int down_reduction_blocks = S1_D_FF / 4;
    constexpr int down_output_blocks = S1_D_MODEL / 16;
    __m512i down_acc[down_output_blocks];
    for (int ob = 0; ob < down_output_blocks; ob++) {
        down_acc[ob] = s1_corrected_accumulator(down_sums + ob * 16);
    }
#pragma GCC unroll 32
    for (int rb = 0; rb < down_reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, hq_shifted + rb * 4, 4);
        __m512i activation = _mm512_set1_epi32((int)activation4);
        for (int ob = 0; ob < down_output_blocks; ob += 4) {
            const size_t offset0 =
                ((size_t)rb * down_output_blocks + ob) * 64;
            const size_t offset1 = offset0 + 64;
            const size_t offset2 = offset0 + 128;
            const size_t offset3 = offset0 + 192;
            const __m512i weight0 = s1_load_weight(down + offset0);
            const __m512i weight1 = s1_load_weight(down + offset1);
            const __m512i weight2 = s1_load_weight(down + offset2);
            const __m512i weight3 = s1_load_weight(down + offset3);
            down_acc[ob] = _mm512_dpbusd_epi32(
                down_acc[ob], activation, weight0);
            down_acc[ob + 1] = _mm512_dpbusd_epi32(
                down_acc[ob + 1], activation, weight1);
            down_acc[ob + 2] = _mm512_dpbusd_epi32(
                down_acc[ob + 2], activation, weight2);
            down_acc[ob + 3] = _mm512_dpbusd_epi32(
                down_acc[ob + 3], activation, weight3);
        }
    }
    const __m512 output_scale = _mm512_set1_ps(s_h * s_down);
    for (int ob = 0; ob < down_output_blocks; ob++) {
        __m512 result =
            _mm512_mul_ps(_mm512_cvtepi32_ps(down_acc[ob]), output_scale);
        _mm512_storeu_ps(out + ob * 16, result);
    }
}

__attribute__((noinline)) static void s2_gate_up_dot_tile(
    const int8_t* gate, const int8_t* up, const int32_t* gate_sums,
    const int32_t* up_sums, const uint8_t* xq_shifted, int32_t* gate_out,
    int32_t* up_out) {
    __m512i acc_gate[S2_GATE_TILE_BLOCKS];
    __m512i acc_up[S2_GATE_TILE_BLOCKS];
#pragma GCC unroll 8
    for (int local_ob = 0; local_ob < S2_GATE_TILE_BLOCKS; local_ob++) {
        acc_gate[local_ob] =
            s1_corrected_accumulator(gate_sums + local_ob * 16);
        acc_up[local_ob] =
            s1_corrected_accumulator(up_sums + local_ob * 16);
    }

    constexpr int reduction_blocks = S2_D_MODEL / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, xq_shifted + rb * 4, 4);
        const __m512i activation = _mm512_set1_epi32((int)activation4);
#pragma GCC unroll 4
        for (int local_ob = 0; local_ob < S2_GATE_TILE_BLOCKS;
             local_ob += 2) {
            const size_t offset0 =
                ((size_t)rb * S2_GATE_TILE_BLOCKS + local_ob) * 64;
            const size_t offset1 = offset0 + 64;
            acc_gate[local_ob] = s2_dpbusd_mem(
                acc_gate[local_ob], activation, gate + offset0);
            acc_up[local_ob] = s2_dpbusd_mem(
                acc_up[local_ob], activation, up + offset0);
            acc_gate[local_ob + 1] = s2_dpbusd_mem(
                acc_gate[local_ob + 1], activation, gate + offset1);
            acc_up[local_ob + 1] = s2_dpbusd_mem(
                acc_up[local_ob + 1], activation, up + offset1);
        }
    }

#pragma GCC unroll 8
    for (int local_ob = 0; local_ob < S2_GATE_TILE_BLOCKS; local_ob++) {
        _mm512_store_si512((__m512i*)(gate_out + local_ob * 16),
                           acc_gate[local_ob]);
        _mm512_store_si512((__m512i*)(up_out + local_ob * 16),
                           acc_up[local_ob]);
    }
}

__attribute__((noinline)) static void s2_down_dot_tile(
    const int8_t* down, const int32_t* down_sums,
    const uint8_t* hq_shifted, int32_t* down_out) {
    __m512i down_acc[S2_DOWN_TILE_BLOCKS];
#pragma GCC unroll 16
    for (int local_ob = 0; local_ob < S2_DOWN_TILE_BLOCKS; local_ob++) {
        down_acc[local_ob] =
            s1_corrected_accumulator(down_sums + local_ob * 16);
    }

    constexpr int reduction_blocks = S2_D_FF / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, hq_shifted + rb * 4, 4);
        const __m512i activation = _mm512_set1_epi32((int)activation4);
#pragma GCC unroll 4
        for (int local_ob = 0; local_ob < S2_DOWN_TILE_BLOCKS;
             local_ob += 4) {
            const size_t offset0 =
                ((size_t)rb * S2_DOWN_TILE_BLOCKS + local_ob) * 64;
            down_acc[local_ob] = s2_dpbusd_mem(
                down_acc[local_ob], activation, down + offset0);
            down_acc[local_ob + 1] = s2_dpbusd_mem(
                down_acc[local_ob + 1], activation, down + offset0 + 64);
            down_acc[local_ob + 2] = s2_dpbusd_mem(
                down_acc[local_ob + 2], activation, down + offset0 + 128);
            down_acc[local_ob + 3] = s2_dpbusd_mem(
                down_acc[local_ob + 3], activation, down + offset0 + 192);
        }
    }

#pragma GCC unroll 16
    for (int local_ob = 0; local_ob < S2_DOWN_TILE_BLOCKS; local_ob++) {
        _mm512_store_si512((__m512i*)(down_out + local_ob * 16),
                           down_acc[local_ob]);
    }
}

static void s2_expert_ffn(int slot, float s_gate, float s_up, float s_down,
                          const uint8_t* xq_shifted, float s_x, float* out) {
    const int8_t* gate =
        s2_gate_packed + (size_t)slot * S2_MATRIX_SIZE;
    const int8_t* up = s2_up_packed + (size_t)slot * S2_MATRIX_SIZE;
    const int8_t* down =
        s2_down_packed + (size_t)slot * S2_MATRIX_SIZE;
    const int32_t* gate_sums = s2_gate_sums + (size_t)slot * S2_D_FF;
    const int32_t* up_sums = s2_up_sums + (size_t)slot * S2_D_FF;
    const int32_t* down_sums =
        s2_down_sums + (size_t)slot * S2_D_MODEL;

    constexpr int gate_reduction_blocks = S2_D_MODEL / 4;
    constexpr int gate_output_blocks = S2_D_FF / 16;
    constexpr int gate_tile_count =
        gate_output_blocks / S2_GATE_TILE_BLOCKS;
    alignas(64) float hidden[S2_D_FF];
    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    const __m512 one = _mm512_set1_ps(1.0f);

    for (int tile = 0; tile < gate_tile_count; tile++) {
        alignas(64) __m512i acc_gate[S2_GATE_TILE_BLOCKS];
        alignas(64) __m512i acc_up[S2_GATE_TILE_BLOCKS];
        const size_t tile_base =
            (size_t)tile * gate_reduction_blocks * S2_GATE_TILE_BLOCKS * 64;
        s2_gate_up_dot_tile(
            gate + tile_base, up + tile_base,
            gate_sums + tile * S2_GATE_TILE_BLOCKS * 16,
            up_sums + tile * S2_GATE_TILE_BLOCKS * 16, xq_shifted,
            (int32_t*)acc_gate, (int32_t*)acc_up);

#pragma GCC unroll 8
        for (int local_ob = 0; local_ob < S2_GATE_TILE_BLOCKS; local_ob++) {
            const int global_ob = tile * S2_GATE_TILE_BLOCKS + local_ob;
            const __m512 vg = _mm512_mul_ps(
                _mm512_cvtepi32_ps(acc_gate[local_ob]), gate_scale);
            const __m512 vu = _mm512_mul_ps(
                _mm512_cvtepi32_ps(acc_up[local_ob]), up_scale);
            const __m512 denominator = _mm512_add_ps(
                one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg)));
            const __m512 silu =
                _mm512_mul_ps(vg, reciprocal512_ps(denominator));
            _mm512_store_ps(hidden + global_ob * 16,
                            _mm512_mul_ps(silu, vu));
        }
    }

    alignas(64) uint8_t hq_shifted[S2_D_FF];
    const float s_h = quantize_s1_u8(hidden, S2_D_FF, hq_shifted);
    constexpr int down_reduction_blocks = S2_D_FF / 4;
    constexpr int down_output_blocks = S2_D_MODEL / 16;
    constexpr int down_tile_count =
        down_output_blocks / S2_DOWN_TILE_BLOCKS;
    const __m512 output_scale = _mm512_set1_ps(s_h * s_down);

    for (int tile = 0; tile < down_tile_count; tile++) {
        alignas(64) __m512i down_acc[S2_DOWN_TILE_BLOCKS];
        const size_t tile_base =
            (size_t)tile * down_reduction_blocks * S2_DOWN_TILE_BLOCKS * 64;
        s2_down_dot_tile(
            down + tile_base,
            down_sums + tile * S2_DOWN_TILE_BLOCKS * 16, hq_shifted,
            (int32_t*)down_acc);

#pragma GCC unroll 16
        for (int local_ob = 0; local_ob < S2_DOWN_TILE_BLOCKS; local_ob++) {
            const int global_ob = tile * S2_DOWN_TILE_BLOCKS + local_ob;
            const __m512 result = _mm512_mul_ps(
                _mm512_cvtepi32_ps(down_acc[local_ob]), output_scale);
            _mm512_storeu_ps(out + global_ob * 16, result);
        }
    }
}

constexpr int S1_MAX_BACKGROUND_WORKERS = MAX_TOP_K;

struct alignas(64) S1WorkerTask {
    int slot;
    float s_gate;
    float s_up;
    float s_down;
    const uint8_t* xq_shifted;
    float s_x;
    float* out;
};

struct alignas(64) S1WorkerCompletion {
    std::atomic<uint64_t> generation{0};
};

class S1WorkerPool {
   public:
    ~S1WorkerPool() {
        stop.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (int i = 0; i < worker_count; i++) {
            if (workers[i].joinable()) workers[i].join();
        }
    }

    void start() {
        if (started) return;
        started = true;

        int max_threads = S1_MAX_BACKGROUND_WORKERS + 1;
        if (const char* value = std::getenv("OMP_NUM_THREADS")) {
            const long configured = std::strtol(value, nullptr, 10);
            if (configured > 0) max_threads = (int)configured;
        }
        worker_count = max_threads - 1;
        if (worker_count < 0) worker_count = 0;
        if (worker_count > S1_MAX_BACKGROUND_WORKERS) {
            worker_count = S1_MAX_BACKGROUND_WORKERS;
        }

        for (int i = 0; i < worker_count; i++) {
            workers[i] = std::thread(&S1WorkerPool::worker_loop, this, i);
        }
        while (ready.load(std::memory_order_acquire) != worker_count) {
            _mm_pause();
        }
#ifdef _OPENMP
        pin_to_openmp_places();
#endif
    }

    int size() const { return worker_count; }

    void set_task(int worker, int slot, float s_gate, float s_up,
                  float s_down, const uint8_t* xq_shifted, float s_x,
                  float* out) {
        tasks[worker] = {slot, s_gate, s_up, s_down, xq_shifted, s_x, out};
    }

    uint64_t launch() {
        const uint64_t current = ++next_generation;
        generation.store(current, std::memory_order_release);
        return current;
    }

    void wait(uint64_t current) const {
        for (int i = 0; i < worker_count; i++) {
            while (completed[i].generation.load(std::memory_order_acquire) !=
                   current) {
                _mm_pause();
            }
        }
    }

   private:
#ifdef _OPENMP
    static void pin_to_place(pthread_t thread, int place) {
        const int cpu_count = omp_get_place_num_procs(place);
        if (cpu_count <= 0 || cpu_count > CPU_SETSIZE) return;

        int cpus[CPU_SETSIZE];
        omp_get_place_proc_ids(place, cpus);
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        for (int i = 0; i < cpu_count; i++) CPU_SET(cpus[i], &affinity);
        pthread_setaffinity_np(thread, sizeof(affinity), &affinity);
    }

    void pin_to_openmp_places() {
        const int place_count = omp_get_num_places();
        if (place_count < worker_count + 1) return;

        int main_place = omp_get_place_num();
        if (main_place < 0) main_place = 0;
        pin_to_place(pthread_self(), main_place);
        for (int i = 0; i < worker_count; i++) {
            const int place = (main_place + i + 1) % place_count;
            pin_to_place(workers[i].native_handle(), place);
        }
    }
#endif

    void worker_loop(int worker) {
        uint64_t observed = generation.load(std::memory_order_acquire);
        ready.fetch_add(1, std::memory_order_release);

        while (!stop.load(std::memory_order_acquire)) {
            uint64_t current;
            do {
                current = generation.load(std::memory_order_acquire);
                if (stop.load(std::memory_order_relaxed)) return;
                _mm_pause();
            } while (current == observed);

            observed = current;
            const S1WorkerTask task = tasks[worker];
            s1_expert_ffn(task.slot, task.s_gate, task.s_up, task.s_down,
                          task.xq_shifted, task.s_x, task.out);
            completed[worker].generation.store(current,
                                               std::memory_order_release);
        }
    }

    alignas(64) S1WorkerTask tasks[S1_MAX_BACKGROUND_WORKERS];
    S1WorkerCompletion completed[S1_MAX_BACKGROUND_WORKERS];
    std::thread workers[S1_MAX_BACKGROUND_WORKERS];
    alignas(64) std::atomic<uint64_t> generation{0};
    alignas(64) std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    uint64_t next_generation = 0;
    int worker_count = 0;
    bool started = false;
};

static S1WorkerPool& s1_worker_pool() {
    static S1WorkerPool pool;
    return pool;
}

static void start_s1_worker_pool() { s1_worker_pool().start(); }
#endif

static void moe_forward_optimized_s1(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    if (!s1_preprocessed) {
        moe_forward_generic(x, w, y, num_tokens);
        return;
    }

    __m512 router_acc0 = _mm512_setzero_ps();
    __m512 router_acc1 = _mm512_setzero_ps();
    __m512 router_acc2 = _mm512_setzero_ps();
    __m512 router_acc3 = _mm512_setzero_ps();
    __m512 router_acc4 = _mm512_setzero_ps();
    __m512 router_acc5 = _mm512_setzero_ps();
    __m512 router_acc6 = _mm512_setzero_ps();
    __m512 router_acc7 = _mm512_setzero_ps();
    for (int d = 0; d < S1_D_MODEL; d += 8) {
        router_acc0 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d]),
            _mm512_load_ps(s1_router_transposed + (size_t)d * S1_NUM_EXPERTS),
            router_acc0);
        router_acc1 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 1]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 1) * S1_NUM_EXPERTS),
            router_acc1);
        router_acc2 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 2]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 2) * S1_NUM_EXPERTS),
            router_acc2);
        router_acc3 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 3]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 3) * S1_NUM_EXPERTS),
            router_acc3);
        router_acc4 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 4]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 4) * S1_NUM_EXPERTS),
            router_acc4);
        router_acc5 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 5]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 5) * S1_NUM_EXPERTS),
            router_acc5);
        router_acc6 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 6]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 6) * S1_NUM_EXPERTS),
            router_acc6);
        router_acc7 = _mm512_fmadd_ps(
            _mm512_set1_ps(x[d + 7]),
            _mm512_load_ps(s1_router_transposed +
                           (size_t)(d + 7) * S1_NUM_EXPERTS),
            router_acc7);
    }
    __m512 router_acc = _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(router_acc0, router_acc1),
                      _mm512_add_ps(router_acc2, router_acc3)),
        _mm512_add_ps(_mm512_add_ps(router_acc4, router_acc5),
                      _mm512_add_ps(router_acc6, router_acc7)));

    const __m512 affinity_vector = reciprocal512_ps(_mm512_add_ps(
        _mm512_set1_ps(1.0f),
        exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), router_acc))));
    alignas(64) float affinity[S1_NUM_EXPERTS];
    _mm512_store_ps(affinity, affinity_vector);

    int topk_idx[MAX_TOP_K];
    const __m512 selection_scores =
        _mm512_add_ps(affinity_vector, _mm512_loadu_ps(w.bias));
    const __m512 negative_infinity = _mm512_set1_ps(-3.402823466e+38F);
    __mmask16 available = 0xffff;
    for (int k = 0; k < w.top_k; k++) {
        const __m512 candidates =
            _mm512_mask_mov_ps(negative_infinity, available, selection_scores);
        const float best_score = _mm512_reduce_max_ps(candidates);
        const __mmask16 matches =
            available &
            _mm512_cmp_ps_mask(candidates, _mm512_set1_ps(best_score),
                               _CMP_EQ_OQ);
        const int best = __builtin_ctz((unsigned)matches);
        topk_idx[k] = best;
        available =
            (__mmask16)(available & (__mmask16)~(1u << best));
    }

    float gate_sum = 0.0f;
    for (int k = 0; k < w.top_k; k++) {
        gate_sum += affinity[topk_idx[k]];
    }

    alignas(64) uint8_t xq_shifted[S1_D_MODEL];
    const float s_x = quantize_s1_u8(x, S1_D_MODEL, xq_shifted);
    alignas(64) float expert_out[MAX_TOP_K + 1][S1_D_MODEL];
    S1WorkerPool& pool = s1_worker_pool();
    const int background_workers = pool.size();
    for (int worker = 0; worker < background_workers; worker++) {
        const int expert = topk_idx[worker];
        pool.set_task(worker, expert + 1, w.s_gate[expert], w.s_up[expert],
                      w.s_down[expert], xq_shifted, s_x,
                      expert_out[worker + 1]);
    }
    const uint64_t generation = pool.launch();

    s1_expert_ffn(0, w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq_shifted, s_x,
                  expert_out[0]);
    for (int k = background_workers; k < w.top_k; k++) {
        const int expert = topk_idx[k];
        s1_expert_ffn(expert + 1, w.s_gate[expert], w.s_up[expert],
                      w.s_down[expert], xq_shifted, s_x, expert_out[k + 1]);
    }
    pool.wait(generation);

    for (int d = 0; d < S1_D_MODEL; d += 16) {
        _mm512_storeu_ps(
            y + d,
            _mm512_add_ps(_mm512_loadu_ps(x + d),
                          _mm512_load_ps(expert_out[0] + d)));
    }

    for (int k = 0; k < w.top_k; k++) {
        const int expert = topk_idx[k];
        const float gate = affinity[expert] / gate_sum;
        const __m512 gate_vector = _mm512_set1_ps(gate);
        for (int d = 0; d < S1_D_MODEL; d += 16) {
            _mm512_storeu_ps(
                y + d,
                _mm512_fmadd_ps(
                    gate_vector, _mm512_load_ps(expert_out[k + 1] + d),
                    _mm512_loadu_ps(y + d)));
        }
    }
#else
    moe_forward_generic(x, w, y, num_tokens);
#endif
}

static void moe_forward_optimized_s2(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    if (!s2_preprocessed) {
        moe_forward_generic(x, w, y, num_tokens);
        return;
    }

    __m512 router_acc[8] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (int d = 0; d < S2_D_MODEL; d += 8) {
#pragma GCC unroll 8
        for (int u = 0; u < 8; u++) {
            router_acc[u] = _mm512_fmadd_ps(
                _mm512_set1_ps(x[d + u]),
                _mm512_load_ps(s2_router_transposed +
                               (size_t)(d + u) * S2_NUM_EXPERTS),
                router_acc[u]);
        }
    }
    const __m512 router_sum0 =
        _mm512_add_ps(router_acc[0], router_acc[1]);
    const __m512 router_sum1 =
        _mm512_add_ps(router_acc[2], router_acc[3]);
    const __m512 router_sum2 =
        _mm512_add_ps(router_acc[4], router_acc[5]);
    const __m512 router_sum3 =
        _mm512_add_ps(router_acc[6], router_acc[7]);
    const __m512 router_vector = _mm512_add_ps(
        _mm512_add_ps(router_sum0, router_sum1),
        _mm512_add_ps(router_sum2, router_sum3));

    const __m512 affinity_vector = reciprocal512_ps(_mm512_add_ps(
        _mm512_set1_ps(1.0f),
        exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), router_vector))));
    alignas(64) float affinity[S2_NUM_EXPERTS];
    _mm512_store_ps(affinity, affinity_vector);

    int topk_idx[MAX_TOP_K];
    const __m512 selection_scores =
        _mm512_add_ps(affinity_vector, _mm512_loadu_ps(w.bias));
    const __m512 negative_infinity = _mm512_set1_ps(-3.402823466e+38F);
    __mmask16 available = 0xffff;
    for (int k = 0; k < w.top_k; k++) {
        const __m512 candidates =
            _mm512_mask_mov_ps(negative_infinity, available, selection_scores);
        const float best_score = _mm512_reduce_max_ps(candidates);
        const __mmask16 matches =
            available &
            _mm512_cmp_ps_mask(candidates, _mm512_set1_ps(best_score),
                               _CMP_EQ_OQ);
        const int best = __builtin_ctz((unsigned)matches);
        topk_idx[k] = best;
        available = (__mmask16)(available & (__mmask16)~(1u << best));
    }

    float gate_sum = 0.0f;
    for (int k = 0; k < w.top_k; k++) {
        gate_sum += affinity[topk_idx[k]];
    }

    alignas(64) uint8_t xq_shifted[S2_D_MODEL];
    const float s_x = quantize_s1_u8(x, S2_D_MODEL, xq_shifted);
    alignas(64) float expert_out[MAX_TOP_K + 1][S2_D_MODEL];
#ifdef _OPENMP
    int expert_threads = omp_get_max_threads();
    if (expert_threads > w.top_k + 1) expert_threads = w.top_k + 1;
#else
    constexpr int expert_threads = 1;
#endif
#pragma omp parallel for schedule(static) num_threads(expert_threads)
    for (int task = 0; task < w.top_k + 1; task++) {
        if (task == 0) {
            s2_expert_ffn(0, w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq_shifted,
                          s_x, expert_out[0]);
        } else {
            const int expert = topk_idx[task - 1];
            s2_expert_ffn(expert + 1, w.s_gate[expert], w.s_up[expert],
                          w.s_down[expert], xq_shifted, s_x,
                          expert_out[task]);
        }
    }

    for (int d = 0; d < S2_D_MODEL; d += 16) {
        _mm512_storeu_ps(
            y + d,
            _mm512_add_ps(_mm512_loadu_ps(x + d),
                          _mm512_load_ps(expert_out[0] + d)));
    }

    for (int k = 0; k < w.top_k; k++) {
        const int expert = topk_idx[k];
        const __m512 gate_vector =
            _mm512_set1_ps(affinity[expert] / gate_sum);
        for (int d = 0; d < S2_D_MODEL; d += 16) {
            _mm512_storeu_ps(
                y + d,
                _mm512_fmadd_ps(
                    gate_vector, _mm512_load_ps(expert_out[k + 1] + d),
                    _mm512_loadu_ps(y + d)));
        }
    }
#else
    moe_forward_generic(x, w, y, num_tokens);
#endif
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
