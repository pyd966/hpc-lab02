// Main task: optimize the MoE forward pass.

#include "moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>

#if defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

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
constexpr int S2_ACTIVE_TASKS = MAX_TOP_K + 1;
constexpr int S2_THREAD_SHARDS = 16;
constexpr int S2_GATE_BLOCKS_PER_SHARD = (S2_D_FF / 16) / S2_THREAD_SHARDS;
constexpr int S2_DOWN_BLOCKS_PER_SHARD =
    (S2_D_MODEL / 16) / S2_THREAD_SHARDS;
constexpr size_t S2_MATRIX_SIZE = (size_t)S2_D_MODEL * S2_D_FF;
constexpr size_t S2_SHARD_MATRIX_SIZE = S2_MATRIX_SIZE / S2_THREAD_SHARDS;
constexpr int S3_NUM_TOKENS = 128;
constexpr int S3_D_MODEL = 256;
constexpr int S3_D_FF = 128;
constexpr int S3_NUM_EXPERTS = 16;
constexpr int S3_EXPERT_SLOTS = S3_NUM_EXPERTS + 1;
constexpr int S3_TOP_K = 4;
constexpr int S3_ROUTED_ASSIGNMENTS = S3_NUM_TOKENS * S3_TOP_K;
constexpr int S3_TILE_ROWS = 16;
constexpr int S3_TILE_BYTES = 64;
constexpr int S3_C_TILES = 6;
constexpr int S3_THREADS = 8;
constexpr int S3_MAX_GROUPED_ROWS =
    S3_ROUTED_ASSIGNMENTS + S3_NUM_EXPERTS * (S3_TILE_ROWS - 1);
constexpr int S3_MAX_TASKS = 2 + 2 * S3_NUM_EXPERTS;
constexpr size_t S3_MATRIX_SIZE = (size_t)S3_D_MODEL * S3_D_FF;

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
static void start_s2_worker_pool();
alignas(64) static float s2_router_transposed[S2_D_MODEL * S2_NUM_EXPERTS];
struct alignas(64) S2ExpertShard {
    alignas(64) int8_t gate[S2_SHARD_MATRIX_SIZE];
    alignas(64) int8_t up[S2_SHARD_MATRIX_SIZE];
    alignas(64) int8_t down[S2_SHARD_MATRIX_SIZE];
    alignas(64) int32_t gate_sums[S2_GATE_BLOCKS_PER_SHARD * 16];
    alignas(64) int32_t up_sums[S2_GATE_BLOCKS_PER_SHARD * 16];
    alignas(64) int32_t down_sums[S2_DOWN_BLOCKS_PER_SHARD * 16];
};

static_assert(sizeof(S2ExpertShard) * S2_EXPERT_SLOTS < 2 * 1024 * 1024,
              "one thread's S2 weights must fit in its private L2");
alignas(64) static S2ExpertShard
    s2_expert_shards[S2_THREAD_SHARDS][S2_EXPERT_SLOTS];
static bool s2_preprocessed = false;

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
alignas(64) static float
    s3_router_transposed[S3_D_MODEL * S3_NUM_EXPERTS];
alignas(64) static int8_t
    s3_gate_packed[S3_EXPERT_SLOTS * S3_MATRIX_SIZE];
alignas(64) static int8_t
    s3_up_packed[S3_EXPERT_SLOTS * S3_MATRIX_SIZE];
alignas(64) static int8_t
    s3_down_packed[S3_EXPERT_SLOTS * S3_MATRIX_SIZE];
static bool s3_preprocessed = false;

struct alignas(64) S3TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[8];
    uint8_t reserved_2[16];
    uint8_t rows[8];
    uint8_t reserved_3[8];
};

static_assert(sizeof(S3TileConfig) == 64);

alignas(64) static const S3TileConfig s3_tile_config = {
    1,
    0,
    {},
    {S3_TILE_BYTES, S3_TILE_BYTES, S3_TILE_BYTES, S3_TILE_BYTES,
     S3_TILE_BYTES, S3_TILE_BYTES, S3_TILE_BYTES, S3_TILE_BYTES},
    {},
    {S3_TILE_ROWS, S3_TILE_ROWS, S3_TILE_ROWS, S3_TILE_ROWS,
     S3_TILE_ROWS, S3_TILE_ROWS, S3_TILE_ROWS, S3_TILE_ROWS},
    {}};

static void pack_s3_matrix(const int8_t* src, int output_dim,
                           int reduction_dim, int8_t* packed) {
    const int output_blocks = output_dim / 16;
    const int reduction_blocks = reduction_dim / 64;
    for (int ob = 0; ob < output_blocks; ob++) {
        for (int rb = 0; rb < reduction_blocks; rb++) {
            int8_t* tile =
                packed + ((size_t)ob * reduction_blocks + rb) * 1024;
            for (int kg = 0; kg < 16; kg++) {
                for (int n = 0; n < 16; n++) {
                    for (int k = 0; k < 4; k++) {
                        tile[kg * 64 + n * 4 + k] =
                            src[(size_t)(ob * 16 + n) * reduction_dim +
                                rb * 64 + kg * 4 + k];
                    }
                }
            }
        }
    }
}

static bool request_s3_amx_permission() {
#if defined(__linux__) && defined(SYS_arch_prctl) && \
    defined(ARCH_REQ_XCOMP_PERM)
    constexpr unsigned long xfeature_xtiledata = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                   xfeature_xtiledata) == 0;
#else
    return true;
#endif
}
#endif

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

enum class S2Projection { Gate, Up, Down };

static void pack_s2_matrix(const int8_t* src, int output_dim,
                           int reduction_dim, int slot,
                           S2Projection projection) {
    const int output_blocks = output_dim / 16;
    const int reduction_blocks = reduction_dim / 4;
    const int blocks_per_shard = output_blocks / S2_THREAD_SHARDS;

    for (int thread = 0; thread < S2_THREAD_SHARDS; thread++) {
        S2ExpertShard& shard = s2_expert_shards[thread][slot];
        int8_t* packed;
        int32_t* row_sums;
        if (projection == S2Projection::Gate) {
            packed = shard.gate;
            row_sums = shard.gate_sums;
        } else if (projection == S2Projection::Up) {
            packed = shard.up;
            row_sums = shard.up_sums;
        } else {
            packed = shard.down;
            row_sums = shard.down_sums;
        }

        const int first_output = thread * blocks_per_shard * 16;
        for (int local_o = 0; local_o < blocks_per_shard * 16; local_o++) {
            const int o = first_output + local_o;
            int32_t sum = 0;
            for (int r = 0; r < reduction_dim; r++) {
                sum += src[(size_t)o * reduction_dim + r];
            }
            row_sums[local_o] = sum;
        }

        for (int rb = 0; rb < reduction_blocks; rb++) {
            for (int local_ob = 0; local_ob < blocks_per_shard; local_ob++) {
                const int ob = thread * blocks_per_shard + local_ob;
                int8_t* block = packed +
                                ((size_t)rb * blocks_per_shard + local_ob) *
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

    pack_s2_matrix(w.sh_gate, S2_D_FF, S2_D_MODEL, 0,
                   S2Projection::Gate);
    pack_s2_matrix(w.sh_up, S2_D_FF, S2_D_MODEL, 0, S2Projection::Up);
    pack_s2_matrix(w.sh_down, S2_D_MODEL, S2_D_FF, 0,
                   S2Projection::Down);

    for (int e = 0; e < S2_NUM_EXPERTS; e++) {
        const int slot = e + 1;
        pack_s2_matrix(w.w_gate + (size_t)e * S2_MATRIX_SIZE, S2_D_FF,
                       S2_D_MODEL, slot, S2Projection::Gate);
        pack_s2_matrix(w.w_up + (size_t)e * S2_MATRIX_SIZE, S2_D_FF,
                       S2_D_MODEL, slot, S2Projection::Up);
        pack_s2_matrix(w.w_down + (size_t)e * S2_MATRIX_SIZE, S2_D_MODEL,
                       S2_D_FF, slot, S2Projection::Down);
    }
    s2_preprocessed = true;
    start_s2_worker_pool();
#endif
}

static void preprocess_s3(MoEWeights& w) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)
    for (int d = 0; d < S3_D_MODEL; d++) {
        for (int e = 0; e < S3_NUM_EXPERTS; e++) {
            s3_router_transposed[(size_t)d * S3_NUM_EXPERTS + e] =
                w.w_router[(size_t)e * S3_D_MODEL + d];
        }
    }

    pack_s3_matrix(w.sh_gate, S3_D_FF, S3_D_MODEL, s3_gate_packed);
    pack_s3_matrix(w.sh_up, S3_D_FF, S3_D_MODEL, s3_up_packed);
    pack_s3_matrix(w.sh_down, S3_D_MODEL, S3_D_FF, s3_down_packed);
    for (int e = 0; e < S3_NUM_EXPERTS; e++) {
        const int slot = e + 1;
        pack_s3_matrix(w.w_gate + (size_t)e * S3_MATRIX_SIZE, S3_D_FF,
                       S3_D_MODEL,
                       s3_gate_packed + (size_t)slot * S3_MATRIX_SIZE);
        pack_s3_matrix(w.w_up + (size_t)e * S3_MATRIX_SIZE, S3_D_FF,
                       S3_D_MODEL,
                       s3_up_packed + (size_t)slot * S3_MATRIX_SIZE);
        pack_s3_matrix(w.w_down + (size_t)e * S3_MATRIX_SIZE, S3_D_MODEL,
                       S3_D_FF,
                       s3_down_packed + (size_t)slot * S3_MATRIX_SIZE);
    }
    s3_preprocessed = request_s3_amx_permission();
#endif
}

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

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
static float quantize_s3_s8(const float* values, int count, int8_t* quantized) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    __m512 max_abs = _mm512_setzero_ps();
    for (int i = 0; i < count; i += 16) {
        const __m512 value = _mm512_loadu_ps(values + i);
        const __m512 absolute = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(value), abs_mask));
        max_abs = _mm512_max_ps(max_abs, absolute);
    }

    const float amax = _mm512_reduce_max_ps(max_abs);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inverse_scale = amax > 0.0f ? 127.0f / amax : 1.0f;
    const __m512 inverse = _mm512_set1_ps(inverse_scale);
    for (int i = 0; i < count; i += 16) {
        const __m512i values_i32 = _mm512_cvtps_epi32(
            _mm512_mul_ps(_mm512_loadu_ps(values + i), inverse));
        _mm_store_si128((__m128i*)(quantized + i),
                        _mm512_cvtsepi32_epi8(values_i32));
    }
    return scale;
}

static inline void s3_tile_zero_outputs(int active) {
    if (active > 0) _tile_zero(2);
    if (active > 1) _tile_zero(3);
    if (active > 2) _tile_zero(4);
    if (active > 3) _tile_zero(5);
    if (active > 4) _tile_zero(6);
    if (active > 5) _tile_zero(7);
}

static inline void s3_tile_accumulate_outputs(
    int active, const int8_t* input, int input_stride, int first_row_tile,
    int reduction_offset) {
    const int8_t* first =
        input + (size_t)first_row_tile * S3_TILE_ROWS * input_stride +
        reduction_offset;
    if (active > 0) {
        _tile_loadd(0, first, input_stride);
        _tile_dpbssd(2, 0, 1);
    }
    if (active > 1) {
        _tile_loadd(0, first + (size_t)S3_TILE_ROWS * input_stride,
                    input_stride);
        _tile_dpbssd(3, 0, 1);
    }
    if (active > 2) {
        _tile_loadd(0, first + (size_t)2 * S3_TILE_ROWS * input_stride,
                    input_stride);
        _tile_dpbssd(4, 0, 1);
    }
    if (active > 3) {
        _tile_loadd(0, first + (size_t)3 * S3_TILE_ROWS * input_stride,
                    input_stride);
        _tile_dpbssd(5, 0, 1);
    }
    if (active > 4) {
        _tile_loadd(0, first + (size_t)4 * S3_TILE_ROWS * input_stride,
                    input_stride);
        _tile_dpbssd(6, 0, 1);
    }
    if (active > 5) {
        _tile_loadd(0, first + (size_t)5 * S3_TILE_ROWS * input_stride,
                    input_stride);
        _tile_dpbssd(7, 0, 1);
    }
}

static inline void s3_tile_store_projection(
    int active, int32_t* output, int output_stride, int first_row_tile,
    int output_offset) {
    int32_t* first =
        output + (size_t)first_row_tile * S3_TILE_ROWS * output_stride +
        output_offset;
    const int stride_bytes = output_stride * (int)sizeof(int32_t);
    if (active > 0) _tile_stored(2, first, stride_bytes);
    if (active > 1)
        _tile_stored(3, first + (size_t)S3_TILE_ROWS * output_stride,
                     stride_bytes);
    if (active > 2)
        _tile_stored(4, first + (size_t)2 * S3_TILE_ROWS * output_stride,
                     stride_bytes);
    if (active > 3)
        _tile_stored(5, first + (size_t)3 * S3_TILE_ROWS * output_stride,
                     stride_bytes);
    if (active > 4)
        _tile_stored(6, first + (size_t)4 * S3_TILE_ROWS * output_stride,
                     stride_bytes);
    if (active > 5)
        _tile_stored(7, first + (size_t)5 * S3_TILE_ROWS * output_stride,
                     stride_bytes);
}

static void s3_amx_projection(const int8_t* input, int rows,
                              int reduction_dim, const int8_t* weights,
                              int output_dim, int32_t* output) {
    const int row_tiles = rows / S3_TILE_ROWS;
    const int output_blocks = output_dim / 16;
    const int reduction_blocks = reduction_dim / S3_TILE_BYTES;
    for (int first_row_tile = 0; first_row_tile < row_tiles;
         first_row_tile += S3_C_TILES) {
        const int active =
            std::min(S3_C_TILES, row_tiles - first_row_tile);
        for (int ob = 0; ob < output_blocks; ob++) {
            s3_tile_zero_outputs(active);
            for (int rb = 0; rb < reduction_blocks; rb++) {
                const int8_t* weight_tile =
                    weights +
                    ((size_t)ob * reduction_blocks + rb) * 1024;
                _tile_loadd(1, weight_tile, S3_TILE_BYTES);
                s3_tile_accumulate_outputs(
                    active, input, reduction_dim, first_row_tile,
                    rb * S3_TILE_BYTES);
            }
            s3_tile_store_projection(active, output, output_dim,
                                     first_row_tile, ob * 16);
        }
    }
}

static inline void s3_tile_store_down(
    int active, int32_t output[S3_C_TILES][S3_TILE_ROWS][16]) {
    if (active > 0) _tile_stored(2, output[0], 64);
    if (active > 1) _tile_stored(3, output[1], 64);
    if (active > 2) _tile_stored(4, output[2], 64);
    if (active > 3) _tile_stored(5, output[3], 64);
    if (active > 4) _tile_stored(6, output[4], 64);
    if (active > 5) _tile_stored(7, output[5], 64);
}

static void s3_amx_down(const int8_t* hidden, int padded_rows, int valid_rows,
                        const int8_t* weights, const float* hidden_scales,
                        float weight_scale, const int* token_ids,
                        const float* mixtures, const int* topk_ranks,
                        int shared_first_token, float* routed_output,
                        bool shared, const float* x, float* y) {
    const int row_tiles = padded_rows / S3_TILE_ROWS;
    constexpr int output_blocks = S3_D_MODEL / 16;
    constexpr int reduction_blocks = S3_D_FF / S3_TILE_BYTES;
    alignas(64)
        int32_t tile_output[S3_C_TILES][S3_TILE_ROWS][16];

    for (int first_row_tile = 0; first_row_tile < row_tiles;
         first_row_tile += S3_C_TILES) {
        const int active =
            std::min(S3_C_TILES, row_tiles - first_row_tile);
        for (int ob = 0; ob < output_blocks; ob++) {
            s3_tile_zero_outputs(active);
            for (int rb = 0; rb < reduction_blocks; rb++) {
                const int8_t* weight_tile =
                    weights +
                    ((size_t)ob * reduction_blocks + rb) * 1024;
                _tile_loadd(1, weight_tile, S3_TILE_BYTES);
                s3_tile_accumulate_outputs(
                    active, hidden, S3_D_FF, first_row_tile,
                    rb * S3_TILE_BYTES);
            }
            s3_tile_store_down(active, tile_output);

            for (int tile = 0; tile < active; tile++) {
                for (int r = 0; r < S3_TILE_ROWS; r++) {
                    const int row =
                        (first_row_tile + tile) * S3_TILE_ROWS + r;
                    if (row >= valid_rows) continue;
                    const int token =
                        shared ? shared_first_token + row : token_ids[row];
                    float scale = hidden_scales[row] * weight_scale;
                    if (!shared) scale *= mixtures[row];
                    const __m512 result = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_load_si512(
                            (const __m512i*)tile_output[tile][r])),
                        _mm512_set1_ps(scale));
                    if (shared) {
                        float* output =
                            y + (size_t)token * S3_D_MODEL + ob * 16;
                        const float* residual =
                            x + (size_t)token * S3_D_MODEL + ob * 16;
                        _mm512_storeu_ps(
                            output,
                            _mm512_add_ps(_mm512_loadu_ps(residual), result));
                    } else {
                        float* output =
                            routed_output +
                            ((size_t)topk_ranks[row] * S3_NUM_TOKENS + token) *
                                S3_D_MODEL +
                            ob * 16;
                        _mm512_storeu_ps(output, result);
                    }
                }
            }
        }
    }
}

static void s3_router_four(const float* x, float* affinities) {
    __m512 accumulator[4][4];
    for (int token = 0; token < 4; token++) {
        for (int lane = 0; lane < 4; lane++) {
            accumulator[token][lane] = _mm512_setzero_ps();
        }
    }

    for (int d = 0; d < S3_D_MODEL; d += 4) {
        __m512 weight[4];
        for (int lane = 0; lane < 4; lane++) {
            weight[lane] = _mm512_load_ps(
                s3_router_transposed +
                (size_t)(d + lane) * S3_NUM_EXPERTS);
        }
        for (int token = 0; token < 4; token++) {
            const float* input =
                x + (size_t)token * S3_D_MODEL + d;
            for (int lane = 0; lane < 4; lane++) {
                accumulator[token][lane] = _mm512_fmadd_ps(
                    _mm512_set1_ps(input[lane]), weight[lane],
                    accumulator[token][lane]);
            }
        }
    }

    const __m512 one = _mm512_set1_ps(1.0f);
    for (int token = 0; token < 4; token++) {
        const __m512 sum = _mm512_add_ps(
            _mm512_add_ps(accumulator[token][0],
                          accumulator[token][1]),
            _mm512_add_ps(accumulator[token][2],
                          accumulator[token][3]));
        const __m512 affinity = reciprocal512_ps(_mm512_add_ps(
            one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), sum))));
        _mm512_store_ps(affinities + token * S3_NUM_EXPERTS, affinity);
    }
}
#endif


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

static inline void s2_gate_up_shard(
    const S2ExpertShard& shard, int shard_index, float s_gate, float s_up,
    const uint8_t* xq_shifted, float s_x, float* hidden) {
    __m512i acc_gate[S2_GATE_BLOCKS_PER_SHARD];
    __m512i acc_up[S2_GATE_BLOCKS_PER_SHARD];
#pragma GCC unroll 2
    for (int ob = 0; ob < S2_GATE_BLOCKS_PER_SHARD; ob++) {
        acc_gate[ob] =
            s1_corrected_accumulator(shard.gate_sums + ob * 16);
        acc_up[ob] = s1_corrected_accumulator(shard.up_sums + ob * 16);
    }

    constexpr int reduction_blocks = S2_D_MODEL / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, xq_shifted + rb * 4, 4);
        const __m512i activation = _mm512_set1_epi32((int)activation4);
        const size_t offset =
            (size_t)rb * S2_GATE_BLOCKS_PER_SHARD * 64;
        acc_gate[0] =
            s2_dpbusd_mem(acc_gate[0], activation, shard.gate + offset);
        acc_up[0] =
            s2_dpbusd_mem(acc_up[0], activation, shard.up + offset);
        acc_gate[1] =
            s2_dpbusd_mem(acc_gate[1], activation, shard.gate + offset + 64);
        acc_up[1] =
            s2_dpbusd_mem(acc_up[1], activation, shard.up + offset + 64);
    }

    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    const __m512 one = _mm512_set1_ps(1.0f);
    const int first_output =
        shard_index * S2_GATE_BLOCKS_PER_SHARD * 16;
#pragma GCC unroll 2
    for (int ob = 0; ob < S2_GATE_BLOCKS_PER_SHARD; ob++) {
        const __m512 vg =
            _mm512_mul_ps(_mm512_cvtepi32_ps(acc_gate[ob]), gate_scale);
        const __m512 vu =
            _mm512_mul_ps(_mm512_cvtepi32_ps(acc_up[ob]), up_scale);
        const __m512 denominator = _mm512_add_ps(
            one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg)));
        const __m512 silu =
            _mm512_mul_ps(vg, reciprocal512_ps(denominator));
        _mm512_store_ps(hidden + first_output + ob * 16,
                        _mm512_mul_ps(silu, vu));
    }
}

static inline void s2_down_shard(
    const S2ExpertShard& shard, int shard_index, float s_down,
    const uint8_t* hq_shifted, float s_h, float mixture, bool shared,
    const float* x, float* y) {
    __m512i acc[S2_DOWN_BLOCKS_PER_SHARD];
#pragma GCC unroll 4
    for (int ob = 0; ob < S2_DOWN_BLOCKS_PER_SHARD; ob++) {
        acc[ob] = s1_corrected_accumulator(shard.down_sums + ob * 16);
    }

    constexpr int reduction_blocks = S2_D_FF / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, hq_shifted + rb * 4, 4);
        const __m512i activation = _mm512_set1_epi32((int)activation4);
        const size_t offset =
            (size_t)rb * S2_DOWN_BLOCKS_PER_SHARD * 64;
        acc[0] = s2_dpbusd_mem(acc[0], activation, shard.down + offset);
        acc[1] =
            s2_dpbusd_mem(acc[1], activation, shard.down + offset + 64);
        acc[2] =
            s2_dpbusd_mem(acc[2], activation, shard.down + offset + 128);
        acc[3] =
            s2_dpbusd_mem(acc[3], activation, shard.down + offset + 192);
    }

    const __m512 output_scale = _mm512_set1_ps(s_h * s_down);
    const __m512 mixture_vector = _mm512_set1_ps(mixture);
    const int first_output =
        shard_index * S2_DOWN_BLOCKS_PER_SHARD * 16;
#pragma GCC unroll 4
    for (int ob = 0; ob < S2_DOWN_BLOCKS_PER_SHARD; ob++) {
        const int output = first_output + ob * 16;
        const __m512 result =
            _mm512_mul_ps(_mm512_cvtepi32_ps(acc[ob]), output_scale);
        if (shared) {
            _mm512_storeu_ps(
                y + output,
                _mm512_add_ps(_mm512_loadu_ps(x + output), result));
        } else {
            _mm512_storeu_ps(
                y + output,
                _mm512_fmadd_ps(mixture_vector, result,
                                _mm512_loadu_ps(y + output)));
        }
    }
}
static inline void s2_gate_up_shard_pair(
    const S2ExpertShard& first, const S2ExpertShard& second, int shard_index,
    float first_s_gate, float first_s_up, float second_s_gate,
    float second_s_up, const uint8_t* xq_shifted, float s_x,
    float* first_hidden, float* second_hidden) {
    const S2ExpertShard* shards[2] = {&first, &second};
    __m512i acc_gate[2][S2_GATE_BLOCKS_PER_SHARD];
    __m512i acc_up[2][S2_GATE_BLOCKS_PER_SHARD];
#pragma GCC unroll 2
    for (int task = 0; task < 2; task++) {
#pragma GCC unroll 2
        for (int ob = 0; ob < S2_GATE_BLOCKS_PER_SHARD; ob++) {
            acc_gate[task][ob] =
                s1_corrected_accumulator(shards[task]->gate_sums + ob * 16);
            acc_up[task][ob] =
                s1_corrected_accumulator(shards[task]->up_sums + ob * 16);
        }
    }

    constexpr int reduction_blocks = S2_D_MODEL / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        uint32_t activation4;
        std::memcpy(&activation4, xq_shifted + rb * 4, 4);
        const __m512i activation = _mm512_set1_epi32((int)activation4);
        const size_t offset =
            (size_t)rb * S2_GATE_BLOCKS_PER_SHARD * 64;
#pragma GCC unroll 2
        for (int task = 0; task < 2; task++) {
            acc_gate[task][0] = s2_dpbusd_mem(
                acc_gate[task][0], activation, shards[task]->gate + offset);
            acc_up[task][0] = s2_dpbusd_mem(
                acc_up[task][0], activation, shards[task]->up + offset);
            acc_gate[task][1] =
                s2_dpbusd_mem(acc_gate[task][1], activation,
                              shards[task]->gate + offset + 64);
            acc_up[task][1] =
                s2_dpbusd_mem(acc_up[task][1], activation,
                              shards[task]->up + offset + 64);
        }
    }

    const float gate_scales[2] = {first_s_gate, second_s_gate};
    const float up_scales[2] = {first_s_up, second_s_up};
    float* hidden[2] = {first_hidden, second_hidden};
    const __m512 one = _mm512_set1_ps(1.0f);
    const int first_output =
        shard_index * S2_GATE_BLOCKS_PER_SHARD * 16;
#pragma GCC unroll 2
    for (int task = 0; task < 2; task++) {
        const __m512 gate_scale = _mm512_set1_ps(s_x * gate_scales[task]);
        const __m512 up_scale = _mm512_set1_ps(s_x * up_scales[task]);
#pragma GCC unroll 2
        for (int ob = 0; ob < S2_GATE_BLOCKS_PER_SHARD; ob++) {
            const __m512 vg = _mm512_mul_ps(
                _mm512_cvtepi32_ps(acc_gate[task][ob]), gate_scale);
            const __m512 vu = _mm512_mul_ps(
                _mm512_cvtepi32_ps(acc_up[task][ob]), up_scale);
            const __m512 denominator = _mm512_add_ps(
                one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg)));
            const __m512 silu =
                _mm512_mul_ps(vg, reciprocal512_ps(denominator));
            _mm512_store_ps(hidden[task] + first_output + ob * 16,
                            _mm512_mul_ps(silu, vu));
        }
    }
}

static inline void s2_down_shard_pair(
    const S2ExpertShard& first, const S2ExpertShard& second, int shard_index,
    float first_s_down, float second_s_down,
    const uint8_t* first_hq_shifted, const uint8_t* second_hq_shifted,
    float first_s_h, float second_s_h, float first_mixture,
    float second_mixture, bool first_shared, const float* x, float* y) {
    const S2ExpertShard* shards[2] = {&first, &second};
    const uint8_t* hq_shifted[2] = {first_hq_shifted, second_hq_shifted};
    __m512i acc[2][S2_DOWN_BLOCKS_PER_SHARD];
#pragma GCC unroll 2
    for (int task = 0; task < 2; task++) {
#pragma GCC unroll 4
        for (int ob = 0; ob < S2_DOWN_BLOCKS_PER_SHARD; ob++) {
            acc[task][ob] =
                s1_corrected_accumulator(shards[task]->down_sums + ob * 16);
        }
    }

    constexpr int reduction_blocks = S2_D_FF / 4;
    for (int rb = 0; rb < reduction_blocks; rb++) {
        const size_t offset =
            (size_t)rb * S2_DOWN_BLOCKS_PER_SHARD * 64;
#pragma GCC unroll 2
        for (int task = 0; task < 2; task++) {
            uint32_t activation4;
            std::memcpy(&activation4, hq_shifted[task] + rb * 4, 4);
            const __m512i activation = _mm512_set1_epi32((int)activation4);
            acc[task][0] = s2_dpbusd_mem(
                acc[task][0], activation, shards[task]->down + offset);
            acc[task][1] =
                s2_dpbusd_mem(acc[task][1], activation,
                              shards[task]->down + offset + 64);
            acc[task][2] =
                s2_dpbusd_mem(acc[task][2], activation,
                              shards[task]->down + offset + 128);
            acc[task][3] =
                s2_dpbusd_mem(acc[task][3], activation,
                              shards[task]->down + offset + 192);
        }
    }

    const __m512 first_output_scale =
        _mm512_set1_ps(first_s_h * first_s_down);
    const __m512 second_output_scale =
        _mm512_set1_ps(second_s_h * second_s_down);
    const __m512 first_mixture_vector = _mm512_set1_ps(first_mixture);
    const __m512 second_mixture_vector = _mm512_set1_ps(second_mixture);
    const int first_output =
        shard_index * S2_DOWN_BLOCKS_PER_SHARD * 16;
#pragma GCC unroll 4
    for (int ob = 0; ob < S2_DOWN_BLOCKS_PER_SHARD; ob++) {
        const int output = first_output + ob * 16;
        const __m512 first_result =
            _mm512_mul_ps(_mm512_cvtepi32_ps(acc[0][ob]), first_output_scale);
        if (first_shared) {
            _mm512_storeu_ps(
                y + output,
                _mm512_add_ps(_mm512_loadu_ps(x + output), first_result));
        } else {
            _mm512_storeu_ps(
                y + output,
                _mm512_fmadd_ps(first_mixture_vector, first_result,
                                _mm512_loadu_ps(y + output)));
        }

        const __m512 second_result = _mm512_mul_ps(
            _mm512_cvtepi32_ps(acc[1][ob]), second_output_scale);
        _mm512_storeu_ps(
            y + output,
            _mm512_fmadd_ps(second_mixture_vector, second_result,
                            _mm512_loadu_ps(y + output)));
    }
}

struct alignas(64) S2TaskReady {
    std::atomic<int> value{0};
};

struct S2WorkContext {
    const uint8_t* xq_shifted;
    float s_x;
    const int* slots;
    const float* gate_scales;
    const float* up_scales;
    const float* down_scales;
    const float* mixtures;
    float (*hidden)[S2_D_FF];
    uint8_t (*hq_shifted)[S2_D_FF];
    float (*hidden_scale)[16];
    S2TaskReady* task_ready;
    const float* x;
    float* y;
    alignas(64) std::atomic<int> gate_arrived{0};
    alignas(64) std::atomic<int> down_arrived{0};
};

static void s2_execute_worker(int thread, int team_size,
                              S2WorkContext& work) {
    for (int shard_index = thread; shard_index < S2_THREAD_SHARDS;
         shard_index += team_size) {
        int task = 0;
        for (; task + 1 < S2_ACTIVE_TASKS; task += 2) {
            const S2ExpertShard& first =
                s2_expert_shards[shard_index][work.slots[task]];
            const S2ExpertShard& second =
                s2_expert_shards[shard_index][work.slots[task + 1]];
            s2_gate_up_shard_pair(
                first, second, shard_index, work.gate_scales[task],
                work.up_scales[task], work.gate_scales[task + 1],
                work.up_scales[task + 1], work.xq_shifted, work.s_x,
                work.hidden[task], work.hidden[task + 1]);
        }
        if (task < S2_ACTIVE_TASKS) {
            const S2ExpertShard& shard =
                s2_expert_shards[shard_index][work.slots[task]];
            s2_gate_up_shard(
                shard, shard_index, work.gate_scales[task],
                work.up_scales[task], work.xq_shifted, work.s_x,
                work.hidden[task]);
        }
    }

    work.gate_arrived.fetch_add(1, std::memory_order_acq_rel);
    while (work.gate_arrived.load(std::memory_order_acquire) != team_size) {
        _mm_pause();
    }

    for (int task = thread; task < S2_ACTIVE_TASKS; task += team_size) {
        work.hidden_scale[task][0] = quantize_s1_u8(
            work.hidden[task], S2_D_FF, work.hq_shifted[task]);
        work.task_ready[task].value.store(1, std::memory_order_release);
    }

    int task = 0;
    for (; task + 1 < S2_ACTIVE_TASKS; task += 2) {
        while (work.task_ready[task].value.load(
                   std::memory_order_acquire) == 0) {
            _mm_pause();
        }
        while (work.task_ready[task + 1].value.load(
                   std::memory_order_acquire) == 0) {
            _mm_pause();
        }
        for (int shard_index = thread; shard_index < S2_THREAD_SHARDS;
             shard_index += team_size) {
            const S2ExpertShard& first =
                s2_expert_shards[shard_index][work.slots[task]];
            const S2ExpertShard& second =
                s2_expert_shards[shard_index][work.slots[task + 1]];
            s2_down_shard_pair(
                first, second, shard_index, work.down_scales[task],
                work.down_scales[task + 1], work.hq_shifted[task],
                work.hq_shifted[task + 1], work.hidden_scale[task][0],
                work.hidden_scale[task + 1][0], work.mixtures[task],
                work.mixtures[task + 1], task == 0, work.x, work.y);
        }
    }
    if (task < S2_ACTIVE_TASKS) {
        while (work.task_ready[task].value.load(
                   std::memory_order_acquire) == 0) {
            _mm_pause();
        }
        for (int shard_index = thread; shard_index < S2_THREAD_SHARDS;
             shard_index += team_size) {
            const S2ExpertShard& shard =
                s2_expert_shards[shard_index][work.slots[task]];
            s2_down_shard(
                shard, shard_index, work.down_scales[task],
                work.hq_shifted[task], work.hidden_scale[task][0],
                work.mixtures[task], task == 0, work.x, work.y);
        }
    }

    work.down_arrived.fetch_add(1, std::memory_order_acq_rel);
}

class S2WorkerPool {
   public:
    ~S2WorkerPool() {
        stop.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (int i = 0; i < worker_count; i++) {
            if (workers[i].joinable()) workers[i].join();
        }
    }

    void start() {
        if (started) return;
        started = true;

        team_size = S2_THREAD_SHARDS;
#ifdef _OPENMP
        const int max_threads = omp_get_max_threads();
        if (team_size > max_threads) team_size = max_threads;
        const int core_places = omp_get_num_places();
        if (core_places > 0 && team_size > core_places) {
            team_size = core_places;
        }
#else
        if (const char* value = std::getenv("OMP_NUM_THREADS")) {
            const long configured = std::strtol(value, nullptr, 10);
            if (configured > 0 && team_size > configured) {
                team_size = (int)configured;
            }
        }
#endif
        if (team_size < 1) team_size = 1;
        worker_count = team_size - 1;

        for (int i = 0; i < worker_count; i++) {
            workers[i] = std::thread(&S2WorkerPool::worker_loop, this, i + 1);
        }
        while (ready.load(std::memory_order_acquire) != worker_count) {
            _mm_pause();
        }
#ifdef _OPENMP
        pin_to_openmp_places();
#endif
    }

    void run(S2WorkContext& work) {
        work.gate_arrived.store(0, std::memory_order_relaxed);
        work.down_arrived.store(0, std::memory_order_relaxed);
        for (int task = 0; task < S2_ACTIVE_TASKS; task++) {
            work.task_ready[task].value.store(0, std::memory_order_relaxed);
        }

        context.store(&work, std::memory_order_relaxed);
        generation.fetch_add(1, std::memory_order_release);
        s2_execute_worker(0, team_size, work);
        while (work.down_arrived.load(std::memory_order_acquire) != team_size) {
            _mm_pause();
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
        if (place_count < team_size) return;

        int main_place = omp_get_place_num();
        if (main_place < 0) main_place = 0;
        pin_to_place(pthread_self(), main_place);
        for (int i = 0; i < worker_count; i++) {
            const int place = (main_place + i + 1) % place_count;
            pin_to_place(workers[i].native_handle(), place);
        }
    }
#endif

    void worker_loop(int thread) {
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
            S2WorkContext* work = context.load(std::memory_order_relaxed);
            s2_execute_worker(thread, team_size, *work);
        }
    }

    std::thread workers[S2_THREAD_SHARDS - 1];
    alignas(64) std::atomic<S2WorkContext*> context{nullptr};
    alignas(64) std::atomic<uint64_t> generation{0};
    alignas(64) std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    int team_size = 1;
    int worker_count = 0;
    bool started = false;
};

static S2WorkerPool& s2_worker_pool() {
    static S2WorkerPool pool;
    return pool;
}

static void start_s2_worker_pool() { s2_worker_pool().start(); }

constexpr int S1_ACTIVE_BACKGROUND_WORKERS = MAX_TOP_K;
constexpr int S13_BACKGROUND_WORKERS = S3_THREADS - 1;

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
struct S3WorkContext;
static void s3_execute_worker(int thread, S3WorkContext& work);
#endif

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
        pthread_mutex_lock(&extra_mutex);
        s3_enabled = true;
        pthread_cond_broadcast(&extra_condition);
        pthread_mutex_unlock(&extra_mutex);
        generation.fetch_add(1, std::memory_order_release);
        for (int i = 0; i < worker_count; i++) {
            if (workers[i].joinable()) workers[i].join();
        }
        pthread_cond_destroy(&extra_condition);
        pthread_mutex_destroy(&extra_mutex);
    }

    void start() {
        if (started) return;
        started = true;

        worker_count = S13_BACKGROUND_WORKERS;

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

    int size() const { return S1_ACTIVE_BACKGROUND_WORKERS; }

    void set_task(int worker, int slot, float s_gate, float s_up,
                  float s_down, const uint8_t* xq_shifted, float s_x,
                  float* out) {
        tasks[worker] = {slot, s_gate, s_up, s_down, xq_shifted, s_x, out};
    }

    uint64_t launch() {
        mode = Mode::S1;
        const uint64_t current = ++next_generation;
        generation.store(current, std::memory_order_release);
        return current;
    }

    void wait(uint64_t current) const {
        for (int i = 0; i < S1_ACTIVE_BACKGROUND_WORKERS; i++) {
            while (completed[i].generation.load(std::memory_order_acquire) !=
                   current) {
                _mm_pause();
            }
        }
    }

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
    void run_s3(S3WorkContext& work) {
        enable_s3_workers();
        s3_context = &work;
        mode = Mode::S3;
        const uint64_t current = ++next_generation;
        generation.store(current, std::memory_order_release);
        s3_execute_worker(0, work);
        for (int i = 0; i < worker_count; i++) {
            while (completed[i].generation.load(std::memory_order_acquire) !=
                   current) {
                _mm_pause();
            }
        }
    }
#endif

   private:
    enum class Mode { S1, S3 };

    void enable_s3_workers() {
        pthread_mutex_lock(&extra_mutex);
        if (!s3_enabled) {
            s3_enabled = true;
            pthread_cond_broadcast(&extra_condition);
        }
        pthread_mutex_unlock(&extra_mutex);
    }

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

        if (worker >= S1_ACTIVE_BACKGROUND_WORKERS) {
            pthread_mutex_lock(&extra_mutex);
            while (!s3_enabled && !stop.load(std::memory_order_acquire)) {
                pthread_cond_wait(&extra_condition, &extra_mutex);
            }
            pthread_mutex_unlock(&extra_mutex);
            if (stop.load(std::memory_order_acquire)) return;
        }

        while (!stop.load(std::memory_order_acquire)) {
            uint64_t current;
            do {
                current = generation.load(std::memory_order_acquire);
                if (stop.load(std::memory_order_relaxed)) return;
                _mm_pause();
            } while (current == observed);

            observed = current;
            if (mode == Mode::S1) {
                if (worker < S1_ACTIVE_BACKGROUND_WORKERS) {
                    const S1WorkerTask task = tasks[worker];
                    s1_expert_ffn(task.slot, task.s_gate, task.s_up,
                                  task.s_down, task.xq_shifted, task.s_x,
                                  task.out);
                }
#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
            } else {
                s3_execute_worker(worker + 1, *s3_context);
#endif
            }
            completed[worker].generation.store(current,
                                               std::memory_order_release);
        }
    }

    alignas(64) S1WorkerTask tasks[S1_ACTIVE_BACKGROUND_WORKERS];
    S1WorkerCompletion completed[S13_BACKGROUND_WORKERS];
    std::thread workers[S13_BACKGROUND_WORKERS];
    alignas(64) std::atomic<uint64_t> generation{0};
    alignas(64) std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    pthread_mutex_t extra_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t extra_condition = PTHREAD_COND_INITIALIZER;
#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
    S3WorkContext* s3_context = nullptr;
#endif
    uint64_t next_generation = 0;
    int worker_count = 0;
    Mode mode = Mode::S1;
    bool s3_enabled = false;
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

    alignas(64) float hidden[S2_ACTIVE_TASKS][S2_D_FF];
    alignas(64) uint8_t hq_shifted[S2_ACTIVE_TASKS][S2_D_FF];
    alignas(64) float hidden_scale[S2_ACTIVE_TASKS][16] = {};
    S2TaskReady task_ready[S2_ACTIVE_TASKS];
    int slots[S2_ACTIVE_TASKS] = {0};
    float gate_scales[S2_ACTIVE_TASKS] = {w.sh_s_gate};
    float up_scales[S2_ACTIVE_TASKS] = {w.sh_s_up};
    float down_scales[S2_ACTIVE_TASKS] = {w.sh_s_down};
    float mixtures[S2_ACTIVE_TASKS] = {1.0f};

    for (int task = 1; task < S2_ACTIVE_TASKS; task++) {
        const int expert = topk_idx[task - 1];
        slots[task] = expert + 1;
        gate_scales[task] = w.s_gate[expert];
        up_scales[task] = w.s_up[expert];
        down_scales[task] = w.s_down[expert];
        mixtures[task] = affinity[expert] / gate_sum;
    }

    S2WorkContext work;
    work.xq_shifted = xq_shifted;
    work.s_x = s_x;
    work.slots = slots;
    work.gate_scales = gate_scales;
    work.up_scales = up_scales;
    work.down_scales = down_scales;
    work.mixtures = mixtures;
    work.hidden = hidden;
    work.hq_shifted = hq_shifted;
    work.hidden_scale = hidden_scale;
    work.task_ready = task_ready;
    work.x = x;
    work.y = y;
    s2_worker_pool().run(work);
#else
    moe_forward_generic(x, w, y, num_tokens);
#endif
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)
struct S3Task {
    int slot;
    int input_row;
    int valid_rows;
    int padded_rows;
    int expert;
    int shared_first_token;
    int tile_count;
    int cost;
    bool shared;
};

struct alignas(64) S3WorkContext {
    const float* x;
    const MoEWeights* weights;
    float* y;

    alignas(64) float affinities[S3_NUM_TOKENS][S3_NUM_EXPERTS];
    alignas(64) int8_t xq[S3_NUM_TOKENS][S3_D_MODEL];
    alignas(64) float input_scales[S3_NUM_TOKENS];
    int selected_experts[S3_NUM_TOKENS][S3_TOP_K];
    float selected_mixtures[S3_NUM_TOKENS][S3_TOP_K];

    alignas(64) int local_counts[S3_THREADS][S3_NUM_EXPERTS];
    alignas(64) int thread_offsets[S3_THREADS][S3_NUM_EXPERTS];
    int expert_counts[S3_NUM_EXPERTS];
    int expert_offsets[S3_NUM_EXPERTS + 1];

    alignas(64) int8_t grouped_xq[S3_MAX_GROUPED_ROWS][S3_D_MODEL];
    alignas(64) float grouped_scales[S3_MAX_GROUPED_ROWS];
    alignas(64) int grouped_tokens[S3_MAX_GROUPED_ROWS];
    alignas(64) int grouped_ranks[S3_MAX_GROUPED_ROWS];
    alignas(64) float grouped_mixtures[S3_MAX_GROUPED_ROWS];

    alignas(64)
        float routed_output[S3_TOP_K][S3_NUM_TOKENS][S3_D_MODEL];

    S3Task tasks[S3_MAX_TASKS];
    int task_count;
    int thread_task_count[S3_THREADS];
    int thread_tasks[S3_THREADS][S3_MAX_TASKS];

    alignas(64) std::atomic<int> route_arrived{0};
    alignas(64) std::atomic<int> offsets_ready{0};
    alignas(64) std::atomic<int> scatter_arrived{0};
    alignas(64) std::atomic<int> expert_arrived{0};
};

static inline void s3_barrier(std::atomic<int>& arrived) {
    arrived.fetch_add(1, std::memory_order_acq_rel);
    while (arrived.load(std::memory_order_acquire) != S3_THREADS) {
        _mm_pause();
    }
}

static void s3_execute_expert(
    int slot, const int8_t* input, int padded_rows, int valid_rows,
    const float* input_scales, float gate_scale, float up_scale,
    float down_scale, const int* token_ids, const float* mixtures,
    const int* topk_ranks, int shared_first_token, float* routed_output,
    bool shared, const float* x, float* y) {
    alignas(64)
        int32_t gate_acc[S3_NUM_TOKENS * S3_D_FF];
    alignas(64)
        int32_t up_acc[S3_NUM_TOKENS * S3_D_FF];
    alignas(64) float hidden[S3_NUM_TOKENS * S3_D_FF];
    alignas(64) int8_t hidden_q[S3_NUM_TOKENS * S3_D_FF];
    alignas(64) float hidden_scales[S3_NUM_TOKENS];

    const int8_t* gate =
        s3_gate_packed + (size_t)slot * S3_MATRIX_SIZE;
    const int8_t* up = s3_up_packed + (size_t)slot * S3_MATRIX_SIZE;
    const int8_t* down =
        s3_down_packed + (size_t)slot * S3_MATRIX_SIZE;
    s3_amx_projection(input, padded_rows, S3_D_MODEL, gate, S3_D_FF,
                      gate_acc);
    s3_amx_projection(input, padded_rows, S3_D_MODEL, up, S3_D_FF, up_acc);

    const __m512 one = _mm512_set1_ps(1.0f);
    for (int row = 0; row < valid_rows; row++) {
        const __m512 row_gate_scale =
            _mm512_set1_ps(input_scales[row] * gate_scale);
        const __m512 row_up_scale =
            _mm512_set1_ps(input_scales[row] * up_scale);
        for (int f = 0; f < S3_D_FF; f += 16) {
            const size_t offset = (size_t)row * S3_D_FF + f;
            const __m512 vg = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    (const __m512i*)(gate_acc + offset))),
                row_gate_scale);
            const __m512 vu = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    (const __m512i*)(up_acc + offset))),
                row_up_scale);
            const __m512 denominator = _mm512_add_ps(
                one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg)));
            const __m512 silu =
                _mm512_mul_ps(vg, reciprocal512_ps(denominator));
            _mm512_store_ps(hidden + offset, _mm512_mul_ps(silu, vu));
        }
        hidden_scales[row] = quantize_s3_s8(
            hidden + (size_t)row * S3_D_FF, S3_D_FF,
            hidden_q + (size_t)row * S3_D_FF);
    }
    if (padded_rows > valid_rows) {
        std::memset(hidden_q + (size_t)valid_rows * S3_D_FF, 0,
                    (size_t)(padded_rows - valid_rows) * S3_D_FF);
    }

    s3_amx_down(hidden_q, padded_rows, valid_rows, down, hidden_scales,
                down_scale, token_ids, mixtures, topk_ranks,
                shared_first_token, routed_output, shared, x, y);
}

static void s3_build_tasks(S3WorkContext& work) {
    work.task_count = 0;
    auto add_expert = [&](int slot, int expert, int input_row, int rows,
                          bool shared) {
        if (rows == 0) return;
        const int total_tiles =
            (rows + S3_TILE_ROWS - 1) / S3_TILE_ROWS;
        const int rounds =
            (total_tiles + S3_C_TILES - 1) / S3_C_TILES;
        const int tiles_per_round = total_tiles / rounds;
        const int extra_tiles = total_tiles % rounds;
        int first_row = 0;
        for (int round = 0; round < rounds; round++) {
            const int tile_count =
                tiles_per_round + (round < extra_tiles ? 1 : 0);
            const int padded_rows = tile_count * S3_TILE_ROWS;
            const int valid_rows = std::min(rows - first_row, padded_rows);
            S3Task& task = work.tasks[work.task_count++];
            task.slot = slot;
            task.input_row = input_row + first_row;
            task.valid_rows = valid_rows;
            task.padded_rows = padded_rows;
            task.expert = expert;
            task.shared_first_token = shared ? first_row : 0;
            task.tile_count = tile_count;
            task.cost = 256 * (tile_count + 1) + valid_rows * 8;
            task.shared = shared;
            first_row += valid_rows;
        }
    };

    add_expert(0, -1, 0, S3_NUM_TOKENS, true);
    for (int expert = 0; expert < S3_NUM_EXPERTS; expert++) {
        add_expert(expert + 1, expert, work.expert_offsets[expert],
                   work.expert_counts[expert], false);
    }

    std::sort(work.tasks, work.tasks + work.task_count,
              [](const S3Task& first, const S3Task& second) {
                  return first.cost > second.cost;
              });
    int thread_cost[S3_THREADS] = {};
    std::memset(work.thread_task_count, 0, sizeof(work.thread_task_count));
    for (int task = 0; task < work.task_count; task++) {
        int best_thread = 0;
        for (int thread = 1; thread < S3_THREADS; thread++) {
            if (thread_cost[thread] < thread_cost[best_thread]) {
                best_thread = thread;
            }
        }
        const int index = work.thread_task_count[best_thread]++;
        work.thread_tasks[best_thread][index] = task;
        thread_cost[best_thread] += work.tasks[task].cost;
    }
}

static void s3_execute_worker(int thread, S3WorkContext& work) {
    const MoEWeights& w = *work.weights;
    int* local_counts = work.local_counts[thread];
    std::memset(local_counts, 0, S3_NUM_EXPERTS * sizeof(int));

    const __m512 bias = _mm512_loadu_ps(w.bias);
    const __m512 negative_infinity = _mm512_set1_ps(-3.402823466e+38F);
    constexpr int router_blocks = S3_NUM_TOKENS / 4;
    for (int block = thread; block < router_blocks; block += S3_THREADS) {
        const int first_token = block * 4;
        s3_router_four(work.x + (size_t)first_token * S3_D_MODEL,
                       work.affinities[first_token]);
        for (int lane = 0; lane < 4; lane++) {
            const int token = first_token + lane;
            const __m512 affinity = _mm512_load_ps(work.affinities[token]);
            const __m512 selection_scores = _mm512_add_ps(affinity, bias);
            __mmask16 available = 0xffff;
            for (int k = 0; k < S3_TOP_K; k++) {
                const __m512 candidates = _mm512_mask_mov_ps(
                    negative_infinity, available, selection_scores);
                const float best_score = _mm512_reduce_max_ps(candidates);
                const __mmask16 matches =
                    available &
                    _mm512_cmp_ps_mask(candidates,
                                       _mm512_set1_ps(best_score), _CMP_EQ_OQ);
                const int expert = __builtin_ctz((unsigned)matches);
                work.selected_experts[token][k] = expert;
                local_counts[expert]++;
                available = (__mmask16)(
                    available & (__mmask16)~(1u << expert));
            }

            float gate_sum = 0.0f;
            for (int k = 0; k < S3_TOP_K; k++) {
                gate_sum +=
                    work.affinities[token][work.selected_experts[token][k]];
            }
            for (int k = 0; k < S3_TOP_K; k++) {
                work.selected_mixtures[token][k] =
                    work.affinities[token]
                                   [work.selected_experts[token][k]] /
                    gate_sum;
            }
            work.input_scales[token] = quantize_s3_s8(
                work.x + (size_t)token * S3_D_MODEL, S3_D_MODEL,
                work.xq[token]);
        }
    }

    s3_barrier(work.route_arrived);
    if (thread == 0) {
        int grouped_row = 0;
        for (int expert = 0; expert < S3_NUM_EXPERTS; expert++) {
            work.expert_offsets[expert] = grouped_row;
            int count = 0;
            for (int owner = 0; owner < S3_THREADS; owner++) {
                work.thread_offsets[owner][expert] = grouped_row + count;
                count += work.local_counts[owner][expert];
            }
            work.expert_counts[expert] = count;
            const int padded_count =
                (count + S3_TILE_ROWS - 1) & -S3_TILE_ROWS;
            if (padded_count > count) {
                std::memset(work.grouped_xq[grouped_row + count], 0,
                            (size_t)(padded_count - count) * S3_D_MODEL);
            }
            grouped_row += padded_count;
        }
        work.expert_offsets[S3_NUM_EXPERTS] = grouped_row;
        s3_build_tasks(work);
        work.offsets_ready.store(1, std::memory_order_release);
    } else {
        while (work.offsets_ready.load(std::memory_order_acquire) == 0) {
            _mm_pause();
        }
    }

    int cursor[S3_NUM_EXPERTS];
    std::memcpy(cursor, work.thread_offsets[thread], sizeof(cursor));
    for (int block = thread; block < router_blocks; block += S3_THREADS) {
        const int first_token = block * 4;
        for (int lane = 0; lane < 4; lane++) {
            const int token = first_token + lane;
            for (int k = 0; k < S3_TOP_K; k++) {
                const int expert = work.selected_experts[token][k];
                const int row = cursor[expert]++;
                std::memcpy(work.grouped_xq[row], work.xq[token],
                            S3_D_MODEL);
                work.grouped_scales[row] = work.input_scales[token];
                work.grouped_tokens[row] = token;
                work.grouped_ranks[row] = k;
                work.grouped_mixtures[row] =
                    work.selected_mixtures[token][k];
            }
        }
    }
    s3_barrier(work.scatter_arrived);

    static thread_local bool amx_ready = request_s3_amx_permission();
    if (!amx_ready) std::abort();
    _tile_loadconfig(&s3_tile_config);
    for (int index = 0; index < work.thread_task_count[thread]; index++) {
        const S3Task& task =
            work.tasks[work.thread_tasks[thread][index]];
        if (task.shared) {
            s3_execute_expert(
                0, work.xq[task.input_row], task.padded_rows,
                task.valid_rows, work.input_scales + task.input_row,
                w.sh_s_gate, w.sh_s_up, w.sh_s_down, nullptr, nullptr,
                nullptr, task.shared_first_token,
                &work.routed_output[0][0][0], true, work.x, work.y);
        } else {
            const int row = task.input_row;
            const int expert = task.expert;
            s3_execute_expert(
                task.slot, work.grouped_xq[row], task.padded_rows,
                task.valid_rows, work.grouped_scales + row,
                w.s_gate[expert], w.s_up[expert], w.s_down[expert],
                work.grouped_tokens + row, work.grouped_mixtures + row,
                work.grouped_ranks + row, 0,
                &work.routed_output[0][0][0], false, work.x, work.y);
        }
    }
    _tile_release();

    s3_barrier(work.expert_arrived);
    for (int block = thread; block < router_blocks; block += S3_THREADS) {
        const int first_token = block * 4;
        for (int lane = 0; lane < 4; lane++) {
            const int token = first_token + lane;
            for (int d = 0; d < S3_D_MODEL; d += 16) {
                __m512 output = _mm512_loadu_ps(
                    work.y + (size_t)token * S3_D_MODEL + d);
                for (int k = 0; k < S3_TOP_K; k++) {
                    output = _mm512_add_ps(
                        output,
                        _mm512_load_ps(work.routed_output[k][token] + d));
                }
                _mm512_storeu_ps(
                    work.y + (size_t)token * S3_D_MODEL + d, output);
            }
        }
    }
}
#endif

static void moe_forward_optimized_s3(const float* x, const MoEWeights& w,
                                     float* y, int num_tokens) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)
    if (!s3_preprocessed) {
        moe_forward_generic(x, w, y, num_tokens);
        return;
    }

    S3WorkContext work;
    work.x = x;
    work.weights = &w;
    work.y = y;
    s1_worker_pool().run_s3(work);
    return;

#else
    moe_forward_generic(x, w, y, num_tokens);
#endif
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
