#if defined(__riscv)

// RISC-V RVV and SpaceMiT IME implementation for all benchmark cases.

#include "moe.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace {

enum class Case {
    S1,
    S2,
    S3,
    S4,
    Unsupported,
};

Case get_case(const MoEWeights& w, int num_tokens) {
    if (num_tokens == 1 && w.d_model == 256 && w.d_ff == 128 &&
        w.num_experts == 16 && w.top_k == 4) {
        return Case::S1;
    }
    if (num_tokens == 1 && w.d_model == 1024 && w.d_ff == 512 &&
        w.num_experts == 16 && w.top_k == 4) {
        return Case::S2;
    }
    if (num_tokens == 128 && w.d_model == 256 && w.d_ff == 128 &&
        w.num_experts == 16 && w.top_k == 4) {
        return Case::S3;
    }
    if (num_tokens == 1024 && w.d_model == 512 && w.d_ff == 128 &&
        w.num_experts == 512 && w.top_k == 2) {
        return Case::S4;
    }
    return Case::Unsupported;
}

#if defined(__riscv) && defined(__riscv_vector)

constexpr int kNumExperts = 16;
constexpr int kActiveExperts = MAX_TOP_K + 1;
constexpr int kS1DModel = 256;
constexpr int kS2DModel = 1024;
constexpr int kS3Tokens = 128;
constexpr int kS3DModel = 256;
constexpr int kS3Dff = 128;
constexpr int kS3TopK = 4;
constexpr int kS3ExpertSlots = kNumExperts + 1;
constexpr int kS3TileRows = 4;
constexpr int kS3TileK = 8;
constexpr int kS3TileBytes = 32;
constexpr int kS3ResultElements = 16;
constexpr int kS3SharedTasks = 4;
constexpr int kS3SharedRows = kS3Tokens / kS3SharedTasks;
constexpr int kS3MaxGroupedRows =
    kS3Tokens * kS3TopK + kNumExperts * (kS3TileRows - 1);
constexpr size_t kS3MatrixSize = (size_t)kS3DModel * kS3Dff;

alignas(64) float s1_router_transposed[kS1DModel * kNumExperts];
alignas(64) float s2_router_transposed[kS2DModel * kNumExperts];
bool s1_preprocessed = false;
bool s2_preprocessed = false;

alignas(64) int8_t s3_gate_packed[kS3ExpertSlots * kS3MatrixSize];
alignas(64) int8_t s3_up_packed[kS3ExpertSlots * kS3MatrixSize];
alignas(64) int8_t s3_down_packed[kS3ExpertSlots * kS3MatrixSize];
alignas(64) int32_t s3_gate_sums[kS3ExpertSlots * kS3Dff];
alignas(64) int32_t s3_up_sums[kS3ExpertSlots * kS3Dff];
alignas(64) int32_t s3_down_sums[kS3ExpertSlots * kS3DModel];
bool s3_preprocessed = false;

alignas(64) uint8_t s3_quantized_input[kS3Tokens][kS3DModel];
alignas(64) float s3_input_scales[kS3Tokens];
int s3_selected_experts[kS3Tokens][kS3TopK];
float s3_selected_mixtures[kS3Tokens][kS3TopK];
int s3_expert_counts[kNumExperts];
int s3_expert_offsets[kNumExperts + 1];
int s3_grouped_rows[kS3Tokens][kS3TopK];
alignas(64) uint8_t
    s3_grouped_input[kS3MaxGroupedRows][kS3DModel];
alignas(64) float s3_grouped_scales[kS3MaxGroupedRows];
alignas(64) float s3_grouped_output[kS3MaxGroupedRows][kS3DModel];
alignas(64) float s3_shared_output[kS3Tokens][kS3DModel];

constexpr int kS4Tokens = 1024;
constexpr int kS4DModel = 512;
constexpr int kS4Dff = 128;
constexpr int kS4Experts = 512;
constexpr int kS4TopK = 2;
constexpr int kS4RouterCandidates = 40;
constexpr int kS4RouterBlocks = kS4Experts / 16;
constexpr int kS4RouterTasks = 4;
constexpr int kS4RouterRows = kS4Tokens / kS4RouterTasks;
constexpr int kS4SharedTasks = 16;
constexpr int kS4SharedRows = kS4Tokens / kS4SharedTasks;
constexpr int kS4BatchRows = 64;
constexpr int kS4Assignments = kS4Tokens * kS4TopK;
constexpr int kS4MaxGroupedRows =
    kS4Assignments + kS4Experts * (kS3TileRows - 1);
constexpr size_t kS4MatrixSize = (size_t)kS4DModel * kS4Dff;
constexpr size_t kS4RoutedBytes =
    (size_t)kS4Experts * kS4MatrixSize;

int8_t* s4_gate_packed = nullptr;
int8_t* s4_up_packed = nullptr;
int8_t* s4_down_packed = nullptr;
alignas(64) int8_t s4_shared_gate_packed[kS4MatrixSize];
alignas(64) int8_t s4_shared_up_packed[kS4MatrixSize];
alignas(64) int8_t s4_shared_down_packed[kS4MatrixSize];
alignas(64) int8_t s4_router_packed[kS4Experts * kS4DModel];
alignas(64) float s4_router_scales[kS4Experts];
alignas(64) int32_t s4_router_sums[kS4Experts];
alignas(64) int32_t s4_gate_sums[kS4Experts * kS4Dff];
alignas(64) int32_t s4_up_sums[kS4Experts * kS4Dff];
alignas(64) int32_t s4_down_sums[kS4Experts * kS4DModel];
alignas(64) int32_t s4_shared_gate_sums[kS4Dff];
alignas(64) int32_t s4_shared_up_sums[kS4Dff];
alignas(64) int32_t s4_shared_down_sums[kS4DModel];
bool s4_preprocessed = false;

alignas(64) uint8_t s4_quantized_input[kS4Tokens][kS4DModel];
alignas(64) float s4_input_scales[kS4Tokens];
alignas(64) uint8_t
    s4_router_input_packed[kS4RouterTasks][kS4RouterRows * kS4DModel];
alignas(64) int32_t s4_router_acc[kS4Tokens][kS4Experts];
int s4_selected_experts[kS4Tokens][kS4TopK];
float s4_selected_mixtures[kS4Tokens][kS4TopK];
int s4_expert_counts[kS4Experts];
int s4_expert_offsets[kS4Experts + 1];
int s4_grouped_rows[kS4Tokens][kS4TopK];
alignas(64) uint8_t
    s4_grouped_input[kS4MaxGroupedRows][kS4DModel];
alignas(64) float s4_grouped_scales[kS4MaxGroupedRows];
alignas(64) float
    s4_grouped_output[kS4MaxGroupedRows][kS4DModel];
alignas(64) float s4_shared_output[kS4Tokens][kS4DModel];

void transpose_router(const float* source, int d_model, float* transposed) {
    for (int d = 0; d < d_model; ++d) {
        for (int expert = 0; expert < kNumExperts; ++expert) {
            transposed[(size_t)d * kNumExperts + expert] =
                source[(size_t)expert * d_model + d];
        }
    }
}

asm(R"IME(
.macro moe_vmadotus vd, vs1, vs2
  .equ moe_ime_error, 1
  .irp d,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
    .ifc \vd, v\d
      .irp s1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        .ifc \vs1, v\s1
          .irp s2,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
            .ifc \vs2, v\s2
              .insn r CUSTOM_1, 1, 0x71, x\d, x\s1, x\s2
              .equ moe_ime_error, 0
            .endif
          .endr
        .endif
      .endr
    .endif
  .endr
  .if moe_ime_error
    .error "unsupported vmadotus register allocation"
  .endif
.endm
)IME");

inline vint32m2_t ime_vmadotus(vint32m2_t accumulator,
                               vuint8m1_t activation,
                               vint8m1_t weights) {
    asm volatile("moe_vmadotus %[acc], %[act], %[weight]"
                 : [acc] "+vr"(accumulator)
                 : [act] "vr"(activation), [weight] "vr"(weights));
    return accumulator;
}

void pack_s3_matrix(const int8_t* source, int output_dim,
                    int reduction_dim, int8_t* packed,
                    int32_t* row_sums) {
    const int output_blocks = output_dim / 4;
    const int reduction_blocks = reduction_dim / kS3TileK;

    for (int output = 0; output < output_dim; ++output) {
        int32_t sum = 0;
        for (int reduction = 0; reduction < reduction_dim; ++reduction) {
            sum += source[(size_t)output * reduction_dim + reduction];
        }
        row_sums[output] = sum;
    }

    for (int output_block = 0; output_block < output_blocks;
         ++output_block) {
        for (int reduction_block = 0;
             reduction_block < reduction_blocks; ++reduction_block) {
            int8_t* tile =
                packed + ((size_t)output_block * reduction_blocks +
                          reduction_block) *
                             kS3TileBytes;
            for (int output = 0; output < 4; ++output) {
                for (int reduction = 0; reduction < kS3TileK;
                     ++reduction) {
                    tile[output * kS3TileK + reduction] =
                        source[(size_t)(output_block * 4 + output) *
                                   reduction_dim +
                               reduction_block * kS3TileK + reduction];
                }
            }
        }
    }
}

void preprocess_s3(MoEWeights& w) {
    pack_s3_matrix(w.sh_gate, kS3Dff, kS3DModel, s3_gate_packed,
                   s3_gate_sums);
    pack_s3_matrix(w.sh_up, kS3Dff, kS3DModel, s3_up_packed,
                   s3_up_sums);
    pack_s3_matrix(w.sh_down, kS3DModel, kS3Dff, s3_down_packed,
                   s3_down_sums);

    for (int expert = 0; expert < kNumExperts; ++expert) {
        const int slot = expert + 1;
        const size_t source_offset = (size_t)expert * kS3MatrixSize;
        const size_t packed_offset = (size_t)slot * kS3MatrixSize;
        pack_s3_matrix(w.w_gate + source_offset, kS3Dff, kS3DModel,
                       s3_gate_packed + packed_offset,
                       s3_gate_sums + (size_t)slot * kS3Dff);
        pack_s3_matrix(w.w_up + source_offset, kS3Dff, kS3DModel,
                       s3_up_packed + packed_offset,
                       s3_up_sums + (size_t)slot * kS3Dff);
        pack_s3_matrix(w.w_down + source_offset, kS3DModel, kS3Dff,
                       s3_down_packed + packed_offset,
                       s3_down_sums + (size_t)slot * kS3DModel);
    }
    s3_preprocessed = true;
}

void preprocess_s4(MoEWeights& w) {
    if (s4_preprocessed) {
        return;
    }

    s4_gate_packed =
        (int8_t*)std::aligned_alloc(64, kS4RoutedBytes);
    s4_up_packed =
        (int8_t*)std::aligned_alloc(64, kS4RoutedBytes);
    s4_down_packed =
        (int8_t*)std::aligned_alloc(64, kS4RoutedBytes);
    if (s4_gate_packed == nullptr || s4_up_packed == nullptr ||
        s4_down_packed == nullptr) {
        std::abort();
    }

    alignas(64) static int8_t
        quantized_router[kS4Experts * kS4DModel];
    for (int expert = 0; expert < kS4Experts; ++expert) {
        const float* source =
            w.w_router + (size_t)expert * kS4DModel;
        float max_abs = 0.0f;
        for (int d = 0; d < kS4DModel; ++d) {
            const float absolute = fabsf(source[d]);
            if (absolute > max_abs) {
                max_abs = absolute;
            }
        }
        const float scale =
            max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inverse =
            max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        int8_t* destination =
            quantized_router + (size_t)expert * kS4DModel;
        for (int d = 0; d < kS4DModel; ++d) {
            destination[d] =
                (int8_t)lrintf(source[d] * inverse);
        }
        s4_router_scales[expert] = scale;
    }
    pack_s3_matrix(
        quantized_router, kS4Experts, kS4DModel,
        s4_router_packed, s4_router_sums);

    pack_s3_matrix(
        w.sh_gate, kS4Dff, kS4DModel,
        s4_shared_gate_packed, s4_shared_gate_sums);
    pack_s3_matrix(
        w.sh_up, kS4Dff, kS4DModel,
        s4_shared_up_packed, s4_shared_up_sums);
    pack_s3_matrix(
        w.sh_down, kS4DModel, kS4Dff,
        s4_shared_down_packed, s4_shared_down_sums);

#pragma omp parallel for schedule(static)
    for (int expert = 0; expert < kS4Experts; ++expert) {
        const size_t offset = (size_t)expert * kS4MatrixSize;
        pack_s3_matrix(
            w.w_gate + offset, kS4Dff, kS4DModel,
            s4_gate_packed + offset,
            s4_gate_sums + (size_t)expert * kS4Dff);
        pack_s3_matrix(
            w.w_up + offset, kS4Dff, kS4DModel,
            s4_up_packed + offset,
            s4_up_sums + (size_t)expert * kS4Dff);
        pack_s3_matrix(
            w.w_down + offset, kS4DModel, kS4Dff,
            s4_down_packed + offset,
            s4_down_sums + (size_t)expert * kS4DModel);
    }
    s4_preprocessed = true;
}

inline int32_t reduce_i16m8(vint16m8_t values, size_t vl) {
    const size_t scalar_vl = __riscv_vsetvl_e32m1(1);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, scalar_vl);
    const vint32m1_t sum =
        __riscv_vwredsum_vs_i16m8_i32m1(values, zero, vl);
    int32_t result;
    __riscv_vse32_v_i32m1(&result, sum, scalar_vl);
    return result;
}

void rvv_dot_pair(const int8_t* lhs, const int8_t* first_rhs,
                  const int8_t* second_rhs, int count, int32_t& first_sum,
                  int32_t& second_sum) {
    first_sum = 0;
    second_sum = 0;
    for (int offset = 0; offset < count;) {
        const size_t vl = __riscv_vsetvl_e8m4(count - offset);
        const vint8m4_t left =
            __riscv_vle8_v_i8m4(lhs + offset, vl);

        const vint8m4_t first =
            __riscv_vle8_v_i8m4(first_rhs + offset, vl);
        const vint16m8_t first_product =
            __riscv_vwmul_vv_i16m8(left, first, vl);
        first_sum += reduce_i16m8(first_product, vl);

        const vint8m4_t second =
            __riscv_vle8_v_i8m4(second_rhs + offset, vl);
        const vint16m8_t second_product =
            __riscv_vwmul_vv_i16m8(left, second, vl);
        second_sum += reduce_i16m8(second_product, vl);
        offset += (int)vl;
    }
}

int32_t rvv_dot(const int8_t* lhs, const int8_t* rhs, int count) {
    int32_t total = 0;
    for (int offset = 0; offset < count;) {
        const size_t vl = __riscv_vsetvl_e8m4(count - offset);
        const vint8m4_t left =
            __riscv_vle8_v_i8m4(lhs + offset, vl);
        const vint8m4_t right =
            __riscv_vle8_v_i8m4(rhs + offset, vl);
        const vint16m8_t product =
            __riscv_vwmul_vv_i16m8(left, right, vl);
        total += reduce_i16m8(product, vl);
        offset += (int)vl;
    }
    return total;
}

float quantize(const float* values, int count, int8_t* quantized) {
    float max_abs = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float absolute = fabsf(values[i]);
        if (absolute > max_abs) {
            max_abs = absolute;
        }
    }

    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inverse_scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
    for (int i = 0; i < count; ++i) {
        quantized[i] = (int8_t)lrintf(values[i] * inverse_scale);
    }
    return scale;
}

void expert_ffn(const int8_t* gate_weights, const int8_t* up_weights,
                const int8_t* down_weights, float s_gate, float s_up,
                float s_down, const int8_t* quantized_input, float s_x,
                int d_model, int d_ff, float* output) {
    alignas(64) float hidden[MAX_D_FF];
    for (int f = 0; f < d_ff; ++f) {
        int32_t gate_acc;
        int32_t up_acc;
        rvv_dot_pair(quantized_input,
                     gate_weights + (size_t)f * d_model,
                     up_weights + (size_t)f * d_model, d_model, gate_acc,
                     up_acc);
        const float gate_value = (float)gate_acc * (s_x * s_gate);
        const float up_value = (float)up_acc * (s_x * s_up);
        hidden[f] =
            (gate_value / (1.0f + expf(-gate_value))) * up_value;
    }

    alignas(64) int8_t quantized_hidden[MAX_D_FF];
    const float s_h = quantize(hidden, d_ff, quantized_hidden);
    const float output_scale = s_h * s_down;
    for (int d = 0; d < d_model; ++d) {
        const int32_t accumulator =
            rvv_dot(quantized_hidden,
                    down_weights + (size_t)d * d_ff, d_ff);
        output[d] = (float)accumulator * output_scale;
    }
}

void compute_affinities(const float* x, int d_model,
                        const float* router_transposed, float* affinity) {
    for (int first_expert = 0; first_expert < kNumExperts;) {
        const size_t vl =
            __riscv_vsetvl_e32m2(kNumExperts - first_expert);
        vfloat32m2_t accumulator0 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t accumulator1 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t accumulator2 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t accumulator3 =
            __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int d = 0; d < d_model; d += 4) {
            const vfloat32m2_t weights0 = __riscv_vle32_v_f32m2(
                router_transposed +
                    (size_t)d * kNumExperts + first_expert,
                vl);
            const vfloat32m2_t weights1 = __riscv_vle32_v_f32m2(
                router_transposed +
                    (size_t)(d + 1) * kNumExperts + first_expert,
                vl);
            const vfloat32m2_t weights2 = __riscv_vle32_v_f32m2(
                router_transposed +
                    (size_t)(d + 2) * kNumExperts + first_expert,
                vl);
            const vfloat32m2_t weights3 = __riscv_vle32_v_f32m2(
                router_transposed +
                    (size_t)(d + 3) * kNumExperts + first_expert,
                vl);
            accumulator0 = __riscv_vfmacc_vf_f32m2(
                accumulator0, x[d], weights0, vl);
            accumulator1 = __riscv_vfmacc_vf_f32m2(
                accumulator1, x[d + 1], weights1, vl);
            accumulator2 = __riscv_vfmacc_vf_f32m2(
                accumulator2, x[d + 2], weights2, vl);
            accumulator3 = __riscv_vfmacc_vf_f32m2(
                accumulator3, x[d + 3], weights3, vl);
        }

        const vfloat32m2_t sum01 = __riscv_vfadd_vv_f32m2(
            accumulator0, accumulator1, vl);
        const vfloat32m2_t sum23 = __riscv_vfadd_vv_f32m2(
            accumulator2, accumulator3, vl);
        const vfloat32m2_t sum =
            __riscv_vfadd_vv_f32m2(sum01, sum23, vl);
        alignas(64) float logits[kNumExperts];
        __riscv_vse32_v_f32m2(logits, sum, vl);
        for (size_t lane = 0; lane < vl; ++lane) {
            affinity[first_expert + (int)lane] =
                1.0f / (1.0f + expf(-logits[lane]));
        }
        first_expert += (int)vl;
    }
}

void select_top_k(const float* affinity, const float* bias, int* topk_idx) {
    bool used[kNumExperts] = {};
    for (int k = 0; k < MAX_TOP_K; ++k) {
        int best = -1;
        for (int expert = 0; expert < kNumExperts; ++expert) {
            if (used[expert]) {
                continue;
            }
            if (best < 0 ||
                affinity[expert] + bias[expert] >
                    affinity[best] + bias[best]) {
                best = expert;
            }
        }
        used[best] = true;
        topk_idx[k] = best;
    }
}

void combine_shared(const float* x, const float* shared, int count, float* y) {
    for (int i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m4(count - i);
        const vfloat32m4_t x_vector =
            __riscv_vle32_v_f32m4(x + i, vl);
        const vfloat32m4_t shared_vector =
            __riscv_vle32_v_f32m4(shared + i, vl);
        __riscv_vse32_v_f32m4(
            y + i,
            __riscv_vfadd_vv_f32m4(x_vector, shared_vector, vl), vl);
        i += (int)vl;
    }
}

void combine_routed(const float* routed, float mixture, int count, float* y) {
    for (int i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m4(count - i);
        const vfloat32m4_t output =
            __riscv_vle32_v_f32m4(routed + i, vl);
        const vfloat32m4_t current =
            __riscv_vle32_v_f32m4(y + i, vl);
        __riscv_vse32_v_f32m4(
            y + i,
            __riscv_vfmacc_vf_f32m4(current, mixture, output, vl), vl);
        i += (int)vl;
    }
}

void forward_single(const float* x, const MoEWeights& w, float* y,
                    const float* router_transposed) {
    alignas(64) float affinity[kNumExperts];
    compute_affinities(x, w.d_model, router_transposed, affinity);

    int topk_idx[MAX_TOP_K];
    select_top_k(affinity, w.bias, topk_idx);
    float gate_sum = 0.0f;
    for (int k = 0; k < MAX_TOP_K; ++k) {
        gate_sum += affinity[topk_idx[k]];
    }

    alignas(64) int8_t quantized_input[MAX_D_MODEL];
    const float s_x = quantize(x, w.d_model, quantized_input);
    alignas(64) float expert_output[kActiveExperts][MAX_D_MODEL];
    const size_t matrix_size = (size_t)w.d_model * w.d_ff;

#pragma omp parallel for schedule(static)
    for (int task = 0; task < kActiveExperts; ++task) {
        if (task == 0) {
            expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate,
                       w.sh_s_up, w.sh_s_down, quantized_input, s_x,
                       w.d_model, w.d_ff, expert_output[task]);
        } else {
            const int expert = topk_idx[task - 1];
            expert_ffn(w.w_gate + (size_t)expert * matrix_size,
                       w.w_up + (size_t)expert * matrix_size,
                       w.w_down + (size_t)expert * matrix_size,
                       w.s_gate[expert], w.s_up[expert], w.s_down[expert],
                       quantized_input, s_x, w.d_model, w.d_ff,
                       expert_output[task]);
        }
    }

    combine_shared(x, expert_output[0], w.d_model, y);
    for (int task = 1; task < kActiveExperts; ++task) {
        const int expert = topk_idx[task - 1];
        combine_routed(expert_output[task],
                       affinity[expert] / gate_sum, w.d_model, y);
    }
}

vfloat32m2_t s3_exp_vector(vfloat32m2_t value, size_t vl) {
    value = __riscv_vfmin_vf_f32m2(value, 88.3762626647949f, vl);
    value = __riscv_vfmax_vf_f32m2(value, -88.3762626647949f, vl);

    const vfloat32m2_t scaled = __riscv_vfmul_vf_f32m2(
        value, 1.44269504088896341f, vl);
    vint32m2_t exponent =
        __riscv_vfcvt_x_f_v_i32m2(scaled, vl);
    const vfloat32m2_t exponent_f =
        __riscv_vfcvt_f_x_v_f32m2(exponent, vl);

    value = __riscv_vfsub_vv_f32m2(
        value,
        __riscv_vfmul_vf_f32m2(exponent_f, 0.693359375f, vl), vl);
    value = __riscv_vfsub_vv_f32m2(
        value,
        __riscv_vfmul_vf_f32m2(exponent_f, -2.12194440e-4f, vl), vl);
    const vfloat32m2_t squared =
        __riscv_vfmul_vv_f32m2(value, value, vl);

    vfloat32m2_t polynomial =
        __riscv_vfmv_v_f_f32m2(1.9875691500e-4f, vl);
    polynomial = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vv_f32m2(polynomial, value, vl),
        1.3981999507e-3f, vl);
    polynomial = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vv_f32m2(polynomial, value, vl),
        8.3334519073e-3f, vl);
    polynomial = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vv_f32m2(polynomial, value, vl),
        4.1665795894e-2f, vl);
    polynomial = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vv_f32m2(polynomial, value, vl),
        1.6666665459e-1f, vl);
    polynomial = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vv_f32m2(polynomial, value, vl),
        5.0000001201e-1f, vl);
    polynomial = __riscv_vfmadd_vv_f32m2(
        polynomial, squared, value, vl);
    polynomial = __riscv_vfadd_vf_f32m2(polynomial, 1.0f, vl);

    exponent = __riscv_vadd_vx_i32m2(exponent, 127, vl);
    exponent = __riscv_vsll_vx_i32m2(exponent, 23, vl);
    const vfloat32m2_t power =
        __riscv_vreinterpret_v_i32m2_f32m2(exponent);
    return __riscv_vfmul_vv_f32m2(polynomial, power, vl);
}

float quantize_shifted(const float* values, int count, uint8_t* shifted) {
    const size_t full_vl = __riscv_vsetvl_e32m4(count);
    vfloat32m4_t max_vector =
        __riscv_vfmv_v_f_f32m4(0.0f, full_vl);
    for (int i = 0; i < count; i += (int)full_vl) {
        const vfloat32m4_t value =
            __riscv_vle32_v_f32m4(values + i, full_vl);
        max_vector = __riscv_vfmax_vv_f32m4(
            max_vector, __riscv_vfabs_v_f32m4(value, full_vl), full_vl);
    }

    const size_t reduction_vl = __riscv_vsetvl_e32m1(1);
    const vfloat32m1_t zero =
        __riscv_vfmv_v_f_f32m1(0.0f, reduction_vl);
    const vfloat32m1_t maximum =
        __riscv_vfredmax_vs_f32m4_f32m1(max_vector, zero, full_vl);
    alignas(4) float reduced_max[1];
    __riscv_vse32_v_f32m1(reduced_max, maximum, reduction_vl);

    const float max_abs = reduced_max[0];
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inverse_scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
    for (int i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m4(count - i);
        const vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(
            __riscv_vle32_v_f32m4(values + i, vl), inverse_scale, vl);
        const vint32m4_t quantized32 =
            __riscv_vfcvt_x_f_v_i32m4(scaled, vl);
        const vint16m2_t quantized16 = __riscv_vnclip_wx_i16m2(
            quantized32, 0, __RISCV_VXRM_RNE, vl);
        vint8m1_t quantized8 = __riscv_vnclip_wx_i8m1(
            quantized16, 0, __RISCV_VXRM_RNE, vl);
        quantized8 = __riscv_vadd_vx_i8m1(quantized8, -128, vl);
        __riscv_vse8_v_i8m1(
            reinterpret_cast<int8_t*>(shifted + i), quantized8, vl);
        i += (int)vl;
    }
    return scale;
}

void pack_s3_activations(const uint8_t* row_major, int rows,
                         int reduction_dim, uint8_t* packed) {
    const int row_tiles = rows / kS3TileRows;
    const int reduction_blocks = reduction_dim / kS3TileK;
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (int reduction_block = 0;
             reduction_block < reduction_blocks; ++reduction_block) {
            uint8_t* tile =
                packed + ((size_t)row_tile * reduction_blocks +
                          reduction_block) *
                             kS3TileBytes;
            for (int row = 0; row < kS3TileRows; ++row) {
                for (int reduction = 0; reduction < kS3TileK;
                     ++reduction) {
                    tile[row * kS3TileK + reduction] =
                        row_major[(size_t)(row_tile * kS3TileRows + row) *
                                      reduction_dim +
                                  reduction_block * kS3TileK + reduction];
                }
            }
        }
    }
}

void s3_ime_projection(const uint8_t* input, int rows,
                       int reduction_dim, const int8_t* weights,
                       const int32_t* row_sums, int output_dim,
                       int32_t* output) {
    const int row_tiles = rows / kS3TileRows;
    const int reduction_blocks = reduction_dim / kS3TileK;
    const int output_blocks = output_dim / 4;

    for (int first_row_tile = 0; first_row_tile < row_tiles;
         first_row_tile += 6) {
        int active = row_tiles - first_row_tile;
        if (active > 6) {
            active = 6;
        }
        for (int output_block = 0; output_block < output_blocks;
             ++output_block) {
            const size_t initial_vl =
                __riscv_vsetvl_e32m2(kS3ResultElements);
            vint32m2_t accumulator0 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);
            vint32m2_t accumulator1 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);
            vint32m2_t accumulator2 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);
            vint32m2_t accumulator3 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);
            vint32m2_t accumulator4 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);
            vint32m2_t accumulator5 =
                __riscv_vmv_v_x_i32m2(0, initial_vl);

            for (int reduction_block = 0;
                 reduction_block < reduction_blocks; ++reduction_block) {
                const size_t byte_vl =
                    __riscv_vsetvl_e8m1(kS3TileBytes);
                const size_t weight_offset =
                    ((size_t)output_block * reduction_blocks +
                     reduction_block) *
                    kS3TileBytes;
                const vint8m1_t weight = __riscv_vle8_v_i8m1(
                    weights + weight_offset, byte_vl);

                const vuint8m1_t activation0 = __riscv_vle8_v_u8m1(
                    input + ((size_t)first_row_tile * reduction_blocks +
                             reduction_block) *
                                kS3TileBytes,
                    byte_vl);
                accumulator0 =
                    ime_vmadotus(accumulator0, activation0, weight);
                if (active > 1) {
                    const vuint8m1_t activation1 =
                        __riscv_vle8_v_u8m1(
                            input +
                                ((size_t)(first_row_tile + 1) *
                                     reduction_blocks +
                                 reduction_block) *
                                    kS3TileBytes,
                            byte_vl);
                    accumulator1 =
                        ime_vmadotus(accumulator1, activation1, weight);
                }
                if (active > 2) {
                    const vuint8m1_t activation2 =
                        __riscv_vle8_v_u8m1(
                            input +
                                ((size_t)(first_row_tile + 2) *
                                     reduction_blocks +
                                 reduction_block) *
                                    kS3TileBytes,
                            byte_vl);
                    accumulator2 =
                        ime_vmadotus(accumulator2, activation2, weight);
                }
                if (active > 3) {
                    const vuint8m1_t activation3 =
                        __riscv_vle8_v_u8m1(
                            input +
                                ((size_t)(first_row_tile + 3) *
                                     reduction_blocks +
                                 reduction_block) *
                                    kS3TileBytes,
                            byte_vl);
                    accumulator3 =
                        ime_vmadotus(accumulator3, activation3, weight);
                }
                if (active > 4) {
                    const vuint8m1_t activation4 =
                        __riscv_vle8_v_u8m1(
                            input +
                                ((size_t)(first_row_tile + 4) *
                                     reduction_blocks +
                                 reduction_block) *
                                    kS3TileBytes,
                            byte_vl);
                    accumulator4 =
                        ime_vmadotus(accumulator4, activation4, weight);
                }
                if (active > 5) {
                    const vuint8m1_t activation5 =
                        __riscv_vle8_v_u8m1(
                            input +
                                ((size_t)(first_row_tile + 5) *
                                     reduction_blocks +
                                 reduction_block) *
                                    kS3TileBytes,
                            byte_vl);
                    accumulator5 =
                        ime_vmadotus(accumulator5, activation5, weight);
                }
            }

            alignas(64) int32_t result0[kS3ResultElements];
            alignas(64) int32_t result1[kS3ResultElements];
            alignas(64) int32_t result2[kS3ResultElements];
            alignas(64) int32_t result3[kS3ResultElements];
            alignas(64) int32_t result4[kS3ResultElements];
            alignas(64) int32_t result5[kS3ResultElements];
            const size_t result_vl =
                __riscv_vsetvl_e32m2(kS3ResultElements);
            __riscv_vse32_v_i32m2(result0, accumulator0, result_vl);
            if (active > 1) {
                __riscv_vse32_v_i32m2(
                    result1, accumulator1, result_vl);
            }
            if (active > 2) {
                __riscv_vse32_v_i32m2(
                    result2, accumulator2, result_vl);
            }
            if (active > 3) {
                __riscv_vse32_v_i32m2(
                    result3, accumulator3, result_vl);
            }
            if (active > 4) {
                __riscv_vse32_v_i32m2(
                    result4, accumulator4, result_vl);
            }
            if (active > 5) {
                __riscv_vse32_v_i32m2(
                    result5, accumulator5, result_vl);
            }

            for (int tile = 0; tile < active; ++tile) {
                const int32_t* result =
                    tile == 0
                        ? result0
                        : (tile == 1
                               ? result1
                               : (tile == 2
                                      ? result2
                                      : (tile == 3
                                             ? result3
                                             : (tile == 4 ? result4
                                                          : result5))));
                const int row_tile = first_row_tile + tile;
                for (int row = 0; row < kS3TileRows; ++row) {
                    for (int lane = 0; lane < 4; ++lane) {
                        const int output_index =
                            output_block * 4 + lane;
                        const size_t index =
                            (size_t)(row_tile * kS3TileRows + row) *
                                output_dim +
                            output_index;
                        output[index] =
                            result[row * 4 + lane] -
                            128 * row_sums[output_index];
                    }
                }
            }
        }
    }
}

void s3_expert_batch(int slot, const uint8_t* input, int padded_rows,
                     int valid_rows, const float* input_scales,
                     float gate_scale, float up_scale, float down_scale,
                     float* output) {
    alignas(64) uint8_t packed_input[kS3Tokens * kS3DModel];
    alignas(64) int32_t gate_acc[kS3Tokens * kS3Dff];
    alignas(64) int32_t up_acc[kS3Tokens * kS3Dff];
    alignas(64) float hidden[kS3Tokens * kS3Dff];
    alignas(64) uint8_t hidden_q[kS3Tokens * kS3Dff];
    alignas(64) float hidden_scales[kS3Tokens];
    alignas(64) uint8_t packed_hidden[kS3Tokens * kS3Dff];
    alignas(64) int32_t down_acc[kS3Tokens * kS3DModel];

    const size_t matrix_offset = (size_t)slot * kS3MatrixSize;
    const int8_t* gate = s3_gate_packed + matrix_offset;
    const int8_t* up = s3_up_packed + matrix_offset;
    const int8_t* down = s3_down_packed + matrix_offset;
    const int32_t* gate_sums =
        s3_gate_sums + (size_t)slot * kS3Dff;
    const int32_t* up_sums =
        s3_up_sums + (size_t)slot * kS3Dff;
    const int32_t* down_sums =
        s3_down_sums + (size_t)slot * kS3DModel;

    pack_s3_activations(input, padded_rows, kS3DModel, packed_input);
    s3_ime_projection(packed_input, padded_rows, kS3DModel,
                      gate, gate_sums, kS3Dff, gate_acc);
    s3_ime_projection(packed_input, padded_rows, kS3DModel,
                      up, up_sums, kS3Dff, up_acc);

    for (int row = 0; row < valid_rows; ++row) {
        const float row_gate_scale = input_scales[row] * gate_scale;
        const float row_up_scale = input_scales[row] * up_scale;
        const size_t row_offset = (size_t)row * kS3Dff;
        for (int f = 0; f < kS3Dff;) {
            const size_t vl = __riscv_vsetvl_e32m2(kS3Dff - f);
            const vint32m2_t gate_integer =
                __riscv_vle32_v_i32m2(gate_acc + row_offset + f, vl);
            const vint32m2_t up_integer =
                __riscv_vle32_v_i32m2(up_acc + row_offset + f, vl);
            const vfloat32m2_t gate = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(gate_integer, vl),
                row_gate_scale, vl);
            const vfloat32m2_t up = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(up_integer, vl),
                row_up_scale, vl);
            const vfloat32m2_t negative_gate =
                __riscv_vfneg_v_f32m2(gate, vl);
            const vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(
                s3_exp_vector(negative_gate, vl), 1.0f, vl);
            const vfloat32m2_t silu =
                __riscv_vfdiv_vv_f32m2(gate, denominator, vl);
            __riscv_vse32_v_f32m2(
                hidden + row_offset + f,
                __riscv_vfmul_vv_f32m2(silu, up, vl), vl);
            f += (int)vl;
        }
        hidden_scales[row] = quantize_shifted(
            hidden + row_offset, kS3Dff, hidden_q + row_offset);
    }
    for (int row = valid_rows; row < padded_rows; ++row) {
        for (int f = 0; f < kS3Dff; ++f) {
            hidden_q[(size_t)row * kS3Dff + f] = 128;
        }
    }

    pack_s3_activations(hidden_q, padded_rows, kS3Dff, packed_hidden);
    s3_ime_projection(packed_hidden, padded_rows, kS3Dff, down,
                      down_sums, kS3DModel, down_acc);
    for (int row = 0; row < valid_rows; ++row) {
        const float scale = hidden_scales[row] * down_scale;
        const size_t row_offset = (size_t)row * kS3DModel;
        for (int d = 0; d < kS3DModel;) {
            const size_t vl = __riscv_vsetvl_e32m4(kS3DModel - d);
            const vint32m4_t integer = __riscv_vle32_v_i32m4(
                down_acc + row_offset + d, vl);
            const vfloat32m4_t value = __riscv_vfmul_vf_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(integer, vl), scale, vl);
            __riscv_vse32_v_f32m4(output + row_offset + d, value, vl);
            d += (int)vl;
        }
    }
}

void forward_s3(const float* x, const MoEWeights& w, float* y) {
#pragma omp parallel for schedule(static)
    for (int token = 0; token < kS3Tokens; ++token) {
        alignas(64) float affinity[kNumExperts];
        compute_affinities(x + (size_t)token * kS3DModel, kS3DModel,
                           s1_router_transposed, affinity);
        select_top_k(affinity, w.bias, s3_selected_experts[token]);

        float gate_sum = 0.0f;
        for (int k = 0; k < kS3TopK; ++k) {
            gate_sum += affinity[s3_selected_experts[token][k]];
        }
        for (int k = 0; k < kS3TopK; ++k) {
            s3_selected_mixtures[token][k] =
                affinity[s3_selected_experts[token][k]] / gate_sum;
        }
        s3_input_scales[token] = quantize_shifted(
            x + (size_t)token * kS3DModel, kS3DModel,
            s3_quantized_input[token]);
    }

    for (int expert = 0; expert < kNumExperts; ++expert) {
        s3_expert_counts[expert] = 0;
    }
    for (int token = 0; token < kS3Tokens; ++token) {
        for (int k = 0; k < kS3TopK; ++k) {
            ++s3_expert_counts[s3_selected_experts[token][k]];
        }
    }

    int grouped_rows = 0;
    for (int expert = 0; expert < kNumExperts; ++expert) {
        s3_expert_offsets[expert] = grouped_rows;
        grouped_rows +=
            (s3_expert_counts[expert] + kS3TileRows - 1) &
            -kS3TileRows;
    }
    s3_expert_offsets[kNumExperts] = grouped_rows;

    int cursors[kNumExperts];
    for (int expert = 0; expert < kNumExperts; ++expert) {
        cursors[expert] = s3_expert_offsets[expert];
        const int valid_end =
            cursors[expert] + s3_expert_counts[expert];
        const int padded_end = s3_expert_offsets[expert + 1];
        for (int row = valid_end; row < padded_end; ++row) {
            for (int d = 0; d < kS3DModel; ++d) {
                s3_grouped_input[row][d] = 128;
            }
        }
    }

    for (int token = 0; token < kS3Tokens; ++token) {
        for (int k = 0; k < kS3TopK; ++k) {
            const int expert = s3_selected_experts[token][k];
            const int row = cursors[expert]++;
            s3_grouped_rows[token][k] = row;
            s3_grouped_scales[row] = s3_input_scales[token];
            for (int d = 0; d < kS3DModel; ++d) {
                s3_grouped_input[row][d] =
                    s3_quantized_input[token][d];
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < kS3SharedTasks + kNumExperts; ++task) {
        if (task < kS3SharedTasks) {
            const int first_row = task * kS3SharedRows;
            s3_expert_batch(
                0, s3_quantized_input[first_row], kS3SharedRows,
                kS3SharedRows, s3_input_scales + first_row,
                w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                s3_shared_output[first_row]);
        } else {
            const int expert = task - kS3SharedTasks;
            const int valid_rows = s3_expert_counts[expert];
            if (valid_rows == 0) {
                continue;
            }
            const int first_row = s3_expert_offsets[expert];
            const int padded_rows =
                s3_expert_offsets[expert + 1] - first_row;
            s3_expert_batch(
                expert + 1, s3_grouped_input[first_row], padded_rows,
                valid_rows, s3_grouped_scales + first_row,
                w.s_gate[expert], w.s_up[expert], w.s_down[expert],
                s3_grouped_output[first_row]);
        }
    }

#pragma omp parallel for schedule(static)
    for (int token = 0; token < kS3Tokens; ++token) {
        const float* token_x = x + (size_t)token * kS3DModel;
        float* token_y = y + (size_t)token * kS3DModel;
        combine_shared(token_x, s3_shared_output[token], kS3DModel,
                       token_y);
        for (int k = 0; k < kS3TopK; ++k) {
            combine_routed(
                s3_grouped_output[s3_grouped_rows[token][k]],
                s3_selected_mixtures[token][k], kS3DModel, token_y);
        }
    }
}

float s4_exact_router_dot(const float* input, const float* weights) {
    const size_t vl = __riscv_vsetvl_e32m4(kS4DModel);
    vfloat32m4_t accumulator0 =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t accumulator1 =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t accumulator2 =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t accumulator3 =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (int d = 0; d < kS4DModel; d += (int)(4 * vl)) {
        accumulator0 = __riscv_vfmacc_vv_f32m4(
            accumulator0,
            __riscv_vle32_v_f32m4(input + d, vl),
            __riscv_vle32_v_f32m4(weights + d, vl), vl);
        accumulator1 = __riscv_vfmacc_vv_f32m4(
            accumulator1,
            __riscv_vle32_v_f32m4(input + d + vl, vl),
            __riscv_vle32_v_f32m4(weights + d + vl, vl), vl);
        accumulator2 = __riscv_vfmacc_vv_f32m4(
            accumulator2,
            __riscv_vle32_v_f32m4(input + d + 2 * vl, vl),
            __riscv_vle32_v_f32m4(weights + d + 2 * vl, vl), vl);
        accumulator3 = __riscv_vfmacc_vv_f32m4(
            accumulator3,
            __riscv_vle32_v_f32m4(input + d + 3 * vl, vl),
            __riscv_vle32_v_f32m4(weights + d + 3 * vl, vl), vl);
    }

    const vfloat32m4_t sum01 = __riscv_vfadd_vv_f32m4(
        accumulator0, accumulator1, vl);
    const vfloat32m4_t sum23 = __riscv_vfadd_vv_f32m4(
        accumulator2, accumulator3, vl);
    alignas(64) float lanes[kS4DModel];
    __riscv_vse32_v_f32m4(
        lanes, __riscv_vfadd_vv_f32m4(sum01, sum23, vl), vl);
    float result = 0.0f;
    for (size_t lane = 0; lane < vl; ++lane) {
        result += lanes[lane];
    }
    return result;
}

void s4_select_router(const float* input, const int32_t* router_acc,
                      float input_scale, const MoEWeights& w,
                      int* selected, float* mixtures) {
    alignas(64) float approximate_scores[kS4Experts];
    for (int expert = 0; expert < kS4Experts;) {
        const size_t vl =
            __riscv_vsetvl_e32m2(kS4Experts - expert);
        const vint32m2_t integer = __riscv_vle32_v_i32m2(
            router_acc + expert, vl);
        vfloat32m2_t logit = __riscv_vfcvt_f_x_v_f32m2(
            integer, vl);
        logit = __riscv_vfmul_vv_f32m2(
            logit,
            __riscv_vle32_v_f32m2(
                s4_router_scales + expert, vl),
            vl);
        logit = __riscv_vfmul_vf_f32m2(
            logit, input_scale, vl);
        const vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(
            s3_exp_vector(
                __riscv_vfneg_v_f32m2(logit, vl), vl),
            1.0f, vl);
        const vfloat32m2_t affinity = __riscv_vfdiv_vv_f32m2(
            __riscv_vfmv_v_f_f32m2(1.0f, vl), denominator, vl);
        __riscv_vse32_v_f32m2(
            approximate_scores + expert,
            __riscv_vfadd_vv_f32m2(
                affinity,
                __riscv_vle32_v_f32m2(w.bias + expert, vl), vl),
            vl);
        expert += (int)vl;
    }

    int candidates[kS4RouterCandidates];
    int second_candidates[kS4RouterBlocks];
    float block_scores[kS4RouterBlocks];
    for (int block = 0; block < kS4RouterBlocks; ++block) {
        const int first_expert = block * 16;
        int first = first_expert;
        int second = first_expert + 1;
        if (approximate_scores[second] >
            approximate_scores[first]) {
            const int temporary = first;
            first = second;
            second = temporary;
        }
        for (int expert = first_expert + 2;
             expert < first_expert + 16; ++expert) {
            if (approximate_scores[expert] >
                approximate_scores[first]) {
                second = first;
                first = expert;
            } else if (approximate_scores[expert] >
                       approximate_scores[second]) {
                second = expert;
            }
        }
        candidates[block] = first;
        second_candidates[block] = second;
        block_scores[block] = approximate_scores[first];
    }

    constexpr int extra_blocks =
        kS4RouterCandidates - kS4RouterBlocks;
    float top_block_scores[extra_blocks];
    int top_blocks[extra_blocks];
    for (int i = 0; i < extra_blocks; ++i) {
        top_block_scores[i] = -3.402823466e+38F;
        top_blocks[i] = kS4RouterBlocks;
    }
    for (int block = 0; block < kS4RouterBlocks; ++block) {
        int position = extra_blocks;
        for (int i = 0; i < extra_blocks; ++i) {
            if (block_scores[block] > top_block_scores[i]) {
                position = i;
                break;
            }
        }
        if (position == extra_blocks) {
            continue;
        }
        for (int i = extra_blocks - 1; i > position; --i) {
            top_block_scores[i] = top_block_scores[i - 1];
            top_blocks[i] = top_blocks[i - 1];
        }
        top_block_scores[position] = block_scores[block];
        top_blocks[position] = block;
    }
    for (int i = 0; i < extra_blocks; ++i) {
        candidates[kS4RouterBlocks + i] =
            second_candidates[top_blocks[i]];
    }

    float best_scores[kS4TopK] = {
        -3.402823466e+38F, -3.402823466e+38F};
    float best_affinities[kS4TopK] = {};
    selected[0] = selected[1] = kS4Experts;
    for (int candidate = 0;
         candidate < kS4RouterCandidates; ++candidate) {
        const int expert = candidates[candidate];
        const float logit = s4_exact_router_dot(
            input,
            w.w_router + (size_t)expert * kS4DModel);
        const float affinity = 1.0f / (1.0f + expf(-logit));
        const float score = affinity + w.bias[expert];
        for (int k = 0; k < kS4TopK; ++k) {
            if (score > best_scores[k] ||
                (score == best_scores[k] &&
                 expert < selected[k])) {
                for (int move = kS4TopK - 1; move > k; --move) {
                    best_scores[move] = best_scores[move - 1];
                    best_affinities[move] =
                        best_affinities[move - 1];
                    selected[move] = selected[move - 1];
                }
                best_scores[k] = score;
                best_affinities[k] = affinity;
                selected[k] = expert;
                break;
            }
        }
    }

    const float gate_sum =
        best_affinities[0] + best_affinities[1];
    mixtures[0] = best_affinities[0] / gate_sum;
    mixtures[1] = best_affinities[1] / gate_sum;
}

void s4_expert_batch(
    const uint8_t* input, int padded_rows, int valid_rows,
    const float* input_scales, const int8_t* gate,
    const int8_t* up, const int8_t* down,
    const int32_t* gate_sums, const int32_t* up_sums,
    const int32_t* down_sums, float gate_scale,
    float up_scale, float down_scale, float* output) {
    alignas(64) uint8_t
        packed_input[kS4BatchRows * kS4DModel];
    alignas(64) int32_t
        gate_acc[kS4BatchRows * kS4Dff];
    alignas(64) int32_t
        up_acc[kS4BatchRows * kS4Dff];
    alignas(64) float hidden[kS4BatchRows * kS4Dff];
    alignas(64) uint8_t hidden_q[kS4BatchRows * kS4Dff];
    alignas(64) float hidden_scales[kS4BatchRows];
    alignas(64) uint8_t
        packed_hidden[kS4BatchRows * kS4Dff];
    alignas(64) int32_t
        down_acc[kS4BatchRows * kS4DModel];

    pack_s3_activations(
        input, padded_rows, kS4DModel, packed_input);
    s3_ime_projection(
        packed_input, padded_rows, kS4DModel,
        gate, gate_sums, kS4Dff, gate_acc);
    s3_ime_projection(
        packed_input, padded_rows, kS4DModel,
        up, up_sums, kS4Dff, up_acc);

    for (int row = 0; row < valid_rows; ++row) {
        const float row_gate_scale =
            input_scales[row] * gate_scale;
        const float row_up_scale =
            input_scales[row] * up_scale;
        const size_t row_offset = (size_t)row * kS4Dff;
        for (int f = 0; f < kS4Dff;) {
            const size_t vl =
                __riscv_vsetvl_e32m2(kS4Dff - f);
            const vint32m2_t gate_integer =
                __riscv_vle32_v_i32m2(
                    gate_acc + row_offset + f, vl);
            const vint32m2_t up_integer =
                __riscv_vle32_v_i32m2(
                    up_acc + row_offset + f, vl);
            const vfloat32m2_t gate_value =
                __riscv_vfmul_vf_f32m2(
                    __riscv_vfcvt_f_x_v_f32m2(
                        gate_integer, vl),
                    row_gate_scale, vl);
            const vfloat32m2_t up_value =
                __riscv_vfmul_vf_f32m2(
                    __riscv_vfcvt_f_x_v_f32m2(
                        up_integer, vl),
                    row_up_scale, vl);
            const vfloat32m2_t denominator =
                __riscv_vfadd_vf_f32m2(
                    s3_exp_vector(
                        __riscv_vfneg_v_f32m2(
                            gate_value, vl),
                        vl),
                    1.0f, vl);
            const vfloat32m2_t silu =
                __riscv_vfdiv_vv_f32m2(
                    gate_value, denominator, vl);
            __riscv_vse32_v_f32m2(
                hidden + row_offset + f,
                __riscv_vfmul_vv_f32m2(
                    silu, up_value, vl),
                vl);
            f += (int)vl;
        }
        hidden_scales[row] = quantize_shifted(
            hidden + row_offset, kS4Dff,
            hidden_q + row_offset);
    }
    if (padded_rows > valid_rows) {
        std::memset(
            hidden_q + (size_t)valid_rows * kS4Dff, 128,
            (size_t)(padded_rows - valid_rows) * kS4Dff);
    }

    pack_s3_activations(
        hidden_q, padded_rows, kS4Dff, packed_hidden);
    s3_ime_projection(
        packed_hidden, padded_rows, kS4Dff,
        down, down_sums, kS4DModel, down_acc);
    for (int row = 0; row < valid_rows; ++row) {
        const float scale =
            hidden_scales[row] * down_scale;
        const size_t row_offset = (size_t)row * kS4DModel;
        for (int d = 0; d < kS4DModel;) {
            const size_t vl =
                __riscv_vsetvl_e32m4(kS4DModel - d);
            const vint32m4_t integer =
                __riscv_vle32_v_i32m4(
                    down_acc + row_offset + d, vl);
            const vfloat32m4_t value =
                __riscv_vfmul_vf_f32m4(
                    __riscv_vfcvt_f_x_v_f32m4(
                        integer, vl),
                    scale, vl);
            __riscv_vse32_v_f32m4(
                output + row_offset + d, value, vl);
            d += (int)vl;
        }
    }
}

void forward_s4(const float* x, const MoEWeights& w, float* y) {
#pragma omp parallel for schedule(static)
    for (int token = 0; token < kS4Tokens; ++token) {
        s4_input_scales[token] = quantize_shifted(
            x + (size_t)token * kS4DModel, kS4DModel,
            s4_quantized_input[token]);
    }

#pragma omp parallel for schedule(static)
    for (int task = 0; task < kS4RouterTasks; ++task) {
        const int first_row = task * kS4RouterRows;
        uint8_t* packed = s4_router_input_packed[task];
        pack_s3_activations(
            s4_quantized_input[first_row], kS4RouterRows,
            kS4DModel, packed);
        s3_ime_projection(
            packed, kS4RouterRows, kS4DModel,
            s4_router_packed, s4_router_sums, kS4Experts,
            &s4_router_acc[first_row][0]);
    }

#pragma omp parallel for schedule(static)
    for (int token = 0; token < kS4Tokens; ++token) {
        s4_select_router(
            x + (size_t)token * kS4DModel,
            s4_router_acc[token], s4_input_scales[token], w,
            s4_selected_experts[token],
            s4_selected_mixtures[token]);
    }

    std::memset(
        s4_expert_counts, 0, sizeof(s4_expert_counts));
    for (int token = 0; token < kS4Tokens; ++token) {
        for (int k = 0; k < kS4TopK; ++k) {
            ++s4_expert_counts[
                s4_selected_experts[token][k]];
        }
    }

    int grouped_rows = 0;
    for (int expert = 0; expert < kS4Experts; ++expert) {
        s4_expert_offsets[expert] = grouped_rows;
        grouped_rows +=
            (s4_expert_counts[expert] + kS3TileRows - 1) &
            -kS3TileRows;
    }
    s4_expert_offsets[kS4Experts] = grouped_rows;

    int cursors[kS4Experts];
    for (int expert = 0; expert < kS4Experts; ++expert) {
        cursors[expert] = s4_expert_offsets[expert];
        const int valid_end =
            cursors[expert] + s4_expert_counts[expert];
        const int padded_end = s4_expert_offsets[expert + 1];
        if (padded_end > valid_end) {
            std::memset(
                s4_grouped_input[valid_end], 128,
                (size_t)(padded_end - valid_end) * kS4DModel);
        }
    }

    for (int token = 0; token < kS4Tokens; ++token) {
        for (int k = 0; k < kS4TopK; ++k) {
            const int expert =
                s4_selected_experts[token][k];
            const int row = cursors[expert]++;
            s4_grouped_rows[token][k] = row;
            s4_grouped_scales[row] = s4_input_scales[token];
            std::memcpy(
                s4_grouped_input[row],
                s4_quantized_input[token], kS4DModel);
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0;
         task < kS4SharedTasks + kS4Experts; ++task) {
        if (task < kS4SharedTasks) {
            const int first_row = task * kS4SharedRows;
            s4_expert_batch(
                s4_quantized_input[first_row],
                kS4SharedRows, kS4SharedRows,
                s4_input_scales + first_row,
                s4_shared_gate_packed,
                s4_shared_up_packed,
                s4_shared_down_packed,
                s4_shared_gate_sums,
                s4_shared_up_sums,
                s4_shared_down_sums,
                w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                s4_shared_output[first_row]);
            continue;
        }

        const int expert = task - kS4SharedTasks;
        const int rows = s4_expert_counts[expert];
        const int first_row = s4_expert_offsets[expert];
        int consumed = 0;
        while (consumed < rows) {
            int valid_rows = rows - consumed;
            if (valid_rows > kS4BatchRows) {
                valid_rows = kS4BatchRows;
            }
            const int padded_rows =
                (valid_rows + kS3TileRows - 1) &
                -kS3TileRows;
            const size_t weight_offset =
                (size_t)expert * kS4MatrixSize;
            s4_expert_batch(
                s4_grouped_input[first_row + consumed],
                padded_rows, valid_rows,
                s4_grouped_scales + first_row + consumed,
                s4_gate_packed + weight_offset,
                s4_up_packed + weight_offset,
                s4_down_packed + weight_offset,
                s4_gate_sums + (size_t)expert * kS4Dff,
                s4_up_sums + (size_t)expert * kS4Dff,
                s4_down_sums + (size_t)expert * kS4DModel,
                w.s_gate[expert], w.s_up[expert],
                w.s_down[expert],
                s4_grouped_output[first_row + consumed]);
            consumed += valid_rows;
        }
    }

#pragma omp parallel for schedule(static)
    for (int token = 0; token < kS4Tokens; ++token) {
        const size_t token_offset =
            (size_t)token * kS4DModel;
        const float mixture0 =
            s4_selected_mixtures[token][0];
        const float mixture1 =
            s4_selected_mixtures[token][1];
        const float* routed0 =
            s4_grouped_output[
                s4_grouped_rows[token][0]];
        const float* routed1 =
            s4_grouped_output[
                s4_grouped_rows[token][1]];
        for (int d = 0; d < kS4DModel;) {
            const size_t vl =
                __riscv_vsetvl_e32m4(kS4DModel - d);
            vfloat32m4_t result = __riscv_vfadd_vv_f32m4(
                __riscv_vle32_v_f32m4(
                    x + token_offset + d, vl),
                __riscv_vle32_v_f32m4(
                    s4_shared_output[token] + d, vl),
                vl);
            result = __riscv_vfmacc_vf_f32m4(
                result, mixture0,
                __riscv_vle32_v_f32m4(routed0 + d, vl), vl);
            result = __riscv_vfmacc_vf_f32m4(
                result, mixture1,
                __riscv_vle32_v_f32m4(routed1 + d, vl), vl);
            __riscv_vse32_v_f32m4(
                y + token_offset + d, result, vl);
            d += (int)vl;
        }
    }
}

#endif

}  // namespace

void preprocess(MoEWeights& w) {
#if defined(__riscv) && defined(__riscv_vector)
    if (w.d_model == kS1DModel && w.d_ff == 128 &&
        w.num_experts == kNumExperts && w.top_k == MAX_TOP_K) {
        transpose_router(w.w_router, kS1DModel, s1_router_transposed);
        s1_preprocessed = true;
        preprocess_s3(w);
        return;
    }
    if (w.d_model == kS2DModel && w.d_ff == 512 &&
        w.num_experts == kNumExperts && w.top_k == MAX_TOP_K) {
        transpose_router(w.w_router, kS2DModel, s2_router_transposed);
        s2_preprocessed = true;
        return;
    }
    if (w.d_model == kS4DModel && w.d_ff == kS4Dff &&
        w.num_experts == kS4Experts && w.top_k == kS4TopK) {
        preprocess_s4(w);
    }
#else
    (void)w;
#endif
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
#if defined(__riscv) && defined(__riscv_vector)
    switch (get_case(w, num_tokens)) {
        case Case::S1:
            if (s1_preprocessed) {
                forward_single(x, w, y, s1_router_transposed);
                return;
            }
            break;
        case Case::S2:
            if (s2_preprocessed) {
                forward_single(x, w, y, s2_router_transposed);
                return;
            }
            break;
        case Case::S3:
            if (s3_preprocessed) {
                forward_s3(x, w, y);
                return;
            }
            break;
        case Case::S4:
            if (s4_preprocessed) {
                forward_s4(x, w, y);
                return;
            }
            break;
        case Case::Unsupported:
            break;
    }
#else
    (void)get_case(w, num_tokens);
#endif
    moe_forward_ref(x, w, y, num_tokens);
}

#else

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
constexpr int S4_NUM_TOKENS = 1024;
constexpr int S4_D_MODEL = 512;
constexpr int S4_D_FF = 128;
constexpr int S4_NUM_EXPERTS = 512;
constexpr int S4_TOP_K = 2;
constexpr int S4_THREADS = 16;
constexpr int S4_ROUTED_ASSIGNMENTS = S4_NUM_TOKENS * S4_TOP_K;
constexpr int S4_ROUTER_CANDIDATES = 40;
constexpr int S4_AMX_THRESHOLD = 8;
constexpr int S4_AMX_ROWS_PER_TASK = 96;
constexpr int S4_SHARED_ROWS_PER_TASK = 96;
constexpr int S4_MAX_GROUPED_ROWS =
    S4_ROUTED_ASSIGNMENTS + S4_NUM_EXPERTS * (S3_TILE_ROWS - 1);
constexpr int S4_MAX_TASKS =
    16 + S4_NUM_EXPERTS +
    2 * (S4_NUM_TOKENS / S4_AMX_ROWS_PER_TASK + 1);
constexpr size_t S4_MATRIX_SIZE = (size_t)S4_D_MODEL * S4_D_FF;

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
static void start_s4_worker_pool();
static void initialize_s4_context();
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

alignas(64) static int8_t s4_router_packed[S4_NUM_EXPERTS * S4_D_MODEL];
alignas(64) static int32_t s4_router_sums[S4_NUM_EXPERTS];
alignas(64) static float s4_router_scales[S4_NUM_EXPERTS];
static int8_t* s4_gate_packed = nullptr;
static int8_t* s4_up_packed = nullptr;
static int8_t* s4_down_packed = nullptr;
alignas(64) static int32_t
    s4_gate_sums[S4_NUM_EXPERTS * S4_D_FF];
alignas(64) static int32_t s4_up_sums[S4_NUM_EXPERTS * S4_D_FF];
alignas(64) static int32_t
    s4_down_sums[S4_NUM_EXPERTS * S4_D_MODEL];
alignas(64) static int8_t s4_shared_gate_packed[S4_MATRIX_SIZE];
alignas(64) static int8_t s4_shared_up_packed[S4_MATRIX_SIZE];
alignas(64) static int8_t s4_shared_down_packed[S4_MATRIX_SIZE];
static bool s4_preprocessed = false;

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

static void compute_row_sums(const int8_t* matrix, int output_dim,
                             int reduction_dim, int32_t* sums) {
    for (int output = 0; output < output_dim; output++) {
        int32_t sum = 0;
        const int8_t* row = matrix + (size_t)output * reduction_dim;
        for (int reduction = 0; reduction < reduction_dim; reduction++) {
            sum += row[reduction];
        }
        sums[output] = sum;
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

static void preprocess_s4(MoEWeights& w) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)
    if (s4_preprocessed) return;

    constexpr size_t routed_bytes =
        (size_t)S4_NUM_EXPERTS * S4_MATRIX_SIZE;
    s4_gate_packed = (int8_t*)std::aligned_alloc(64, routed_bytes);
    s4_up_packed = (int8_t*)std::aligned_alloc(64, routed_bytes);
    s4_down_packed = (int8_t*)std::aligned_alloc(64, routed_bytes);
    if (!s4_gate_packed || !s4_up_packed || !s4_down_packed) std::abort();

    alignas(64) int8_t quantized_router[S4_NUM_EXPERTS * S4_D_MODEL];
    for (int expert = 0; expert < S4_NUM_EXPERTS; expert++) {
        const float* row = w.w_router + (size_t)expert * S4_D_MODEL;
        float amax = 0.0f;
        for (int d = 0; d < S4_D_MODEL; d++) {
            amax = std::max(amax, std::fabs(row[d]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inverse = amax > 0.0f ? 127.0f / amax : 1.0f;
        int8_t* quantized =
            quantized_router + (size_t)expert * S4_D_MODEL;
        for (int d = 0; d < S4_D_MODEL; d++) {
            quantized[d] = (int8_t)lrintf(row[d] * inverse);
        }
        s4_router_scales[expert] = scale;
    }
    compute_row_sums(quantized_router, S4_NUM_EXPERTS, S4_D_MODEL,
                     s4_router_sums);
    pack_s3_matrix(quantized_router, S4_NUM_EXPERTS, S4_D_MODEL,
                   s4_router_packed);

    pack_s3_matrix(w.sh_gate, S4_D_FF, S4_D_MODEL,
                   s4_shared_gate_packed);
    pack_s3_matrix(w.sh_up, S4_D_FF, S4_D_MODEL, s4_shared_up_packed);
    pack_s3_matrix(w.sh_down, S4_D_MODEL, S4_D_FF,
                   s4_shared_down_packed);

    for (int expert = 0; expert < S4_NUM_EXPERTS; expert++) {
        const size_t offset = (size_t)expert * S4_MATRIX_SIZE;
        const int8_t* gate = w.w_gate + offset;
        const int8_t* up = w.w_up + offset;
        const int8_t* down = w.w_down + offset;
        pack_s3_matrix(gate, S4_D_FF, S4_D_MODEL,
                       s4_gate_packed + offset);
        pack_s3_matrix(up, S4_D_FF, S4_D_MODEL, s4_up_packed + offset);
        pack_s3_matrix(down, S4_D_MODEL, S4_D_FF,
                       s4_down_packed + offset);
        compute_row_sums(gate, S4_D_FF, S4_D_MODEL,
                         s4_gate_sums + (size_t)expert * S4_D_FF);
        compute_row_sums(up, S4_D_FF, S4_D_MODEL,
                         s4_up_sums + (size_t)expert * S4_D_FF);
        compute_row_sums(down, S4_D_MODEL, S4_D_FF,
                         s4_down_sums + (size_t)expert * S4_D_MODEL);
    }

    s4_preprocessed = request_s3_amx_permission();
    if (s4_preprocessed) {
        initialize_s4_context();
        start_s4_worker_pool();
    }
#else
    (void)w;
#endif
}

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
#pragma GCC unroll 32
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
#pragma GCC unroll 16
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
    alignas(64) std::atomic<int> next_task{0};

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
    work.next_task.store(S3_THREADS, std::memory_order_relaxed);
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
    int index = thread;
    for (;;) {
        if (index >= work.task_count) break;
        const S3Task& task = work.tasks[index];
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
        index = work.next_task.fetch_add(1, std::memory_order_relaxed);
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
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)

enum class S4TaskKind { SharedAmx, RoutedAmx, RoutedVnni };

struct S4Task {
    S4TaskKind kind;
    int expert;
    int first_row;
    int valid_rows;
    int padded_rows;
    int cost;
};

struct alignas(64) S4WorkContext {
    const float* x;
    const MoEWeights* weights;
    float* y;

    alignas(64) int8_t xq[S4_NUM_TOKENS][S4_D_MODEL];
    alignas(64) float input_scales[S4_NUM_TOKENS];
    alignas(64)
        int32_t router_acc[S4_NUM_TOKENS][S4_NUM_EXPERTS];
    int selected_experts[S4_NUM_TOKENS][S4_TOP_K];
    float selected_mixtures[S4_NUM_TOKENS][S4_TOP_K];

    alignas(64) int local_counts[S4_THREADS][S4_NUM_EXPERTS];
    alignas(64) int thread_offsets[S4_THREADS][S4_NUM_EXPERTS];
    int expert_counts[S4_NUM_EXPERTS];
    int expert_offsets[S4_NUM_EXPERTS + 1];

    alignas(64) int8_t grouped_xq[S4_MAX_GROUPED_ROWS][S4_D_MODEL];
    alignas(64) float grouped_scales[S4_MAX_GROUPED_ROWS];
    alignas(64) int grouped_tokens[S4_MAX_GROUPED_ROWS];
    alignas(64) int grouped_ranks[S4_MAX_GROUPED_ROWS];
    alignas(64) float grouped_mixtures[S4_MAX_GROUPED_ROWS];

    alignas(64)
        float routed_output[S4_TOP_K][S4_NUM_TOKENS][S4_D_MODEL];

    S4Task tasks[S4_MAX_TASKS];
    int task_count;
    int thread_task_count[S4_THREADS];
    int thread_tasks[S4_THREADS][S4_MAX_TASKS];

    alignas(64) std::atomic<int> quantize_arrived{0};
    alignas(64) std::atomic<int> router_arrived{0};
    alignas(64) std::atomic<int> route_arrived{0};
    alignas(64) std::atomic<int> offsets_ready{0};
    alignas(64) std::atomic<int> scatter_arrived{0};
    alignas(64) std::atomic<int> expert_arrived{0};
};

static void s4_execute_worker(int thread, S4WorkContext& work);

static inline void s4_barrier(std::atomic<int>& arrived) {
    arrived.fetch_add(1, std::memory_order_acq_rel);
    while (arrived.load(std::memory_order_acquire) != S4_THREADS) {
        _mm_pause();
    }
}

static inline const int8_t* s4_vnni_block(const int8_t* packed,
                                          int reduction_dim,
                                          int output_block,
                                          int reduction_block4) {
    const int reduction_tiles = reduction_dim / S3_TILE_BYTES;
    return packed +
           ((size_t)output_block * reduction_tiles +
            reduction_block4 / 16) *
               1024 +
           (reduction_block4 & 15) * 64;
}

static inline __m512i s4_shifted_activation(const int8_t* input,
                                             int reduction_block4) {
    uint32_t word;
    std::memcpy(&word, input + reduction_block4 * 4, sizeof(word));
    return _mm512_set1_epi32((int)(word ^ 0x80808080u));
}

static void s4_vnni_projection(const int8_t* input, const int8_t* weights,
                               const int32_t* row_sums, int reduction_dim,
                               int output_dim, int32_t* output) {
    constexpr int blocks_at_once = 8;
    const int output_blocks = output_dim / 16;
    const int reduction_blocks = reduction_dim / 4;
    for (int first_ob = 0; first_ob < output_blocks;
         first_ob += blocks_at_once) {
        const int active =
            std::min(blocks_at_once, output_blocks - first_ob);
        __m512i accumulator[blocks_at_once];
        for (int local = 0; local < active; local++) {
            accumulator[local] = _mm512_sub_epi32(
                _mm512_setzero_si512(),
                _mm512_slli_epi32(
                    _mm512_load_si512((const __m512i*)(
                        row_sums + (first_ob + local) * 16)),
                    7));
        }
        for (int rb = 0; rb < reduction_blocks; rb++) {
            const __m512i activation = s4_shifted_activation(input, rb);
            for (int local = 0; local < active; local++) {
                accumulator[local] = s2_dpbusd_mem(
                    accumulator[local], activation,
                    s4_vnni_block(weights, reduction_dim,
                                  first_ob + local, rb));
            }
        }
        for (int local = 0; local < active; local++) {
            _mm512_store_si512(
                (__m512i*)(output + (first_ob + local) * 16),
                accumulator[local]);
        }
    }
}

static void s4_router_select(
    const float* x, const int32_t* router_acc, float input_scale,
    const MoEWeights& w, int selected[S4_TOP_K],
    float mixtures[S4_TOP_K]) {
    alignas(64) float approximate_scores[S4_NUM_EXPERTS];
    constexpr int output_blocks = S4_NUM_EXPERTS / 16;

    for (int block = 0; block < output_blocks; block++) {
        const int expert = block * 16;
        const __m512 scale = _mm512_mul_ps(
            _mm512_set1_ps(input_scale),
            _mm512_load_ps(s4_router_scales + expert));
        const __m512 logit = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(
                (const __m512i*)(router_acc + expert))),
            scale);
        const __m512 affinity = reciprocal512_ps(_mm512_add_ps(
            _mm512_set1_ps(1.0f),
            exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), logit))));
        _mm512_store_ps(
            approximate_scores + expert,
            _mm512_add_ps(affinity, _mm512_loadu_ps(w.bias + expert)));
    }

    constexpr int extra_blocks =
        S4_ROUTER_CANDIDATES - output_blocks;
    int candidates[S4_ROUTER_CANDIDATES];
    int second_candidates[output_blocks];
    float block_scores[output_blocks];
    const __m512 negative_infinity =
        _mm512_set1_ps(-3.402823466e+38F);
    for (int block = 0; block < output_blocks; block++) {
        __m512 scores =
            _mm512_load_ps(approximate_scores + block * 16);
        const float first_score = _mm512_reduce_max_ps(scores);
        const __mmask16 first_matches = _mm512_cmp_ps_mask(
            scores, _mm512_set1_ps(first_score), _CMP_EQ_OQ);
        const int first_lane = __builtin_ctz((unsigned)first_matches);
        candidates[block] = block * 16 + first_lane;
        block_scores[block] = first_score;
        scores = _mm512_mask_mov_ps(
            scores, (__mmask16)(1u << first_lane), negative_infinity);
        const float second_score = _mm512_reduce_max_ps(scores);
        const __mmask16 second_matches = _mm512_cmp_ps_mask(
            scores, _mm512_set1_ps(second_score), _CMP_EQ_OQ);
        second_candidates[block] =
            block * 16 + __builtin_ctz((unsigned)second_matches);
    }

    float top_block_scores[extra_blocks];
    int top_blocks[extra_blocks];
    for (int i = 0; i < extra_blocks; i++) {
        top_block_scores[i] = -3.402823466e+38F;
        top_blocks[i] = output_blocks;
    }
    for (int block = 0; block < output_blocks; block++) {
        int position = extra_blocks;
        for (int i = 0; i < extra_blocks; i++) {
            if (block_scores[block] > top_block_scores[i]) {
                position = i;
                break;
            }
        }
        if (position == extra_blocks) continue;
        for (int i = extra_blocks - 1; i > position; i--) {
            top_block_scores[i] = top_block_scores[i - 1];
            top_blocks[i] = top_blocks[i - 1];
        }
        top_block_scores[position] = block_scores[block];
        top_blocks[position] = block;
    }
    for (int i = 0; i < extra_blocks; i++) {
        candidates[output_blocks + i] = second_candidates[top_blocks[i]];
    }

    float best_scores[S4_TOP_K] = {-3.402823466e+38F,
                                   -3.402823466e+38F};
    float best_affinities[S4_TOP_K] = {};
    selected[0] = selected[1] = S4_NUM_EXPERTS;
    for (int candidate = 0; candidate < S4_ROUTER_CANDIDATES;
         candidate += 2) {
        const int experts[2] = {candidates[candidate],
                                candidates[candidate + 1]};
        const float* weights0 =
            w.w_router + (size_t)experts[0] * S4_D_MODEL;
        const float* weights1 =
            w.w_router + (size_t)experts[1] * S4_D_MODEL;
        __m512 accumulator0[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                                 _mm512_setzero_ps(), _mm512_setzero_ps()};
        __m512 accumulator1[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                                 _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int d = 0; d < S4_D_MODEL; d += 64) {
            const __m512 input0 = _mm512_loadu_ps(x + d);
            const __m512 input1 = _mm512_loadu_ps(x + d + 16);
            const __m512 input2 = _mm512_loadu_ps(x + d + 32);
            const __m512 input3 = _mm512_loadu_ps(x + d + 48);
            accumulator0[0] = _mm512_fmadd_ps(
                input0, _mm512_loadu_ps(weights0 + d), accumulator0[0]);
            accumulator0[1] = _mm512_fmadd_ps(
                input1, _mm512_loadu_ps(weights0 + d + 16), accumulator0[1]);
            accumulator0[2] = _mm512_fmadd_ps(
                input2, _mm512_loadu_ps(weights0 + d + 32), accumulator0[2]);
            accumulator0[3] = _mm512_fmadd_ps(
                input3, _mm512_loadu_ps(weights0 + d + 48), accumulator0[3]);
            accumulator1[0] = _mm512_fmadd_ps(
                input0, _mm512_loadu_ps(weights1 + d), accumulator1[0]);
            accumulator1[1] = _mm512_fmadd_ps(
                input1, _mm512_loadu_ps(weights1 + d + 16), accumulator1[1]);
            accumulator1[2] = _mm512_fmadd_ps(
                input2, _mm512_loadu_ps(weights1 + d + 32), accumulator1[2]);
            accumulator1[3] = _mm512_fmadd_ps(
                input3, _mm512_loadu_ps(weights1 + d + 48), accumulator1[3]);
        }
        const __m512 sum010 = _mm512_add_ps(accumulator0[0], accumulator0[1]);
        const __m512 sum230 = _mm512_add_ps(accumulator0[2], accumulator0[3]);
        const __m512 sum011 = _mm512_add_ps(accumulator1[0], accumulator1[1]);
        const __m512 sum231 = _mm512_add_ps(accumulator1[2], accumulator1[3]);
        const float logits[2] = {
            _mm512_reduce_add_ps(_mm512_add_ps(sum010, sum230)),
            _mm512_reduce_add_ps(_mm512_add_ps(sum011, sum231))};
        for (int lane = 0; lane < 2; lane++) {
            const int expert = experts[lane];
            const float logit = logits[lane];
            const float affinity = 1.0f / (1.0f + expf(-logit));
            const float score = affinity + w.bias[expert];

            for (int k = 0; k < S4_TOP_K; k++) {
                if (score > best_scores[k] ||
                    (score == best_scores[k] && expert < selected[k])) {
                    for (int move = S4_TOP_K - 1; move > k; move--) {
                        best_scores[move] = best_scores[move - 1];
                        best_affinities[move] = best_affinities[move - 1];
                        selected[move] = selected[move - 1];
                    }
                    best_scores[k] = score;
                    best_affinities[k] = affinity;
                    selected[k] = expert;
                    break;
                }
            }
        }
    }

    const float gate_sum = best_affinities[0] + best_affinities[1];
    mixtures[0] = best_affinities[0] / gate_sum;
    mixtures[1] = best_affinities[1] / gate_sum;
}

static void s4_vnni_expert(const S4WorkContext& work, int expert, int row) {
    const MoEWeights& w = *work.weights;
    const int8_t* input = work.grouped_xq[row];
    alignas(64) int32_t gate_acc[S4_D_FF];
    alignas(64) int32_t up_acc[S4_D_FF];
    alignas(64) float hidden[S4_D_FF];
    alignas(64) int8_t hidden_q[S4_D_FF];
    alignas(64) int32_t down_acc[S4_D_MODEL];

    const size_t weight_offset = (size_t)expert * S4_MATRIX_SIZE;
    s4_vnni_projection(
        input, s4_gate_packed + weight_offset,
        s4_gate_sums + (size_t)expert * S4_D_FF, S4_D_MODEL,
        S4_D_FF, gate_acc);
    s4_vnni_projection(
        input, s4_up_packed + weight_offset,
        s4_up_sums + (size_t)expert * S4_D_FF, S4_D_MODEL,
        S4_D_FF, up_acc);

    const __m512 gate_scale = _mm512_set1_ps(
        work.grouped_scales[row] * w.s_gate[expert]);
    const __m512 up_scale = _mm512_set1_ps(
        work.grouped_scales[row] * w.s_up[expert]);
    for (int f = 0; f < S4_D_FF; f += 16) {
        const __m512 gate = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_load_si512((const __m512i*)(gate_acc + f))),
            gate_scale);
        const __m512 up = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_load_si512((const __m512i*)(up_acc + f))),
            up_scale);
        const __m512 sigmoid = reciprocal512_ps(_mm512_add_ps(
            _mm512_set1_ps(1.0f),
            exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), gate))));
        _mm512_store_ps(hidden + f,
                        _mm512_mul_ps(_mm512_mul_ps(gate, sigmoid), up));
    }

    const float hidden_scale =
        quantize_s3_s8(hidden, S4_D_FF, hidden_q);
    s4_vnni_projection(
        hidden_q, s4_down_packed + weight_offset,
        s4_down_sums + (size_t)expert * S4_D_MODEL, S4_D_FF,
        S4_D_MODEL, down_acc);

    const int token = work.grouped_tokens[row];
    const int rank = work.grouped_ranks[row];
    float* output =
        const_cast<float*>(&work.routed_output[rank][token][0]);
    const __m512 scale = _mm512_set1_ps(
        hidden_scale * w.s_down[expert] * work.grouped_mixtures[row]);
    for (int d = 0; d < S4_D_MODEL; d += 16) {
        _mm512_store_ps(
            output + d,
            _mm512_mul_ps(
                _mm512_cvtepi32_ps(
                    _mm512_load_si512((const __m512i*)(down_acc + d))),
                scale));
    }
}

static void s4_amx_expert(S4WorkContext& work, const S4Task& task) {
    const MoEWeights& w = *work.weights;
    const bool shared = task.kind == S4TaskKind::SharedAmx;
    const int expert = task.expert;
    const int8_t* input =
        shared ? work.xq[task.first_row] : work.grouped_xq[task.first_row];
    const float* input_scales =
        shared ? work.input_scales + task.first_row
               : work.grouped_scales + task.first_row;
    const int8_t* gate =
        shared ? s4_shared_gate_packed
               : s4_gate_packed + (size_t)expert * S4_MATRIX_SIZE;
    const int8_t* up =
        shared ? s4_shared_up_packed
               : s4_up_packed + (size_t)expert * S4_MATRIX_SIZE;
    const int8_t* down =
        shared ? s4_shared_down_packed
               : s4_down_packed + (size_t)expert * S4_MATRIX_SIZE;
    const float gate_weight_scale =
        shared ? w.sh_s_gate : w.s_gate[expert];
    const float up_weight_scale =
        shared ? w.sh_s_up : w.s_up[expert];
    const float down_weight_scale =
        shared ? w.sh_s_down : w.s_down[expert];

    alignas(64)
        int32_t gate_acc[S4_AMX_ROWS_PER_TASK * S4_D_FF];
    alignas(64)
        int32_t up_acc[S4_AMX_ROWS_PER_TASK * S4_D_FF];
    alignas(64) float hidden[S4_AMX_ROWS_PER_TASK * S4_D_FF];
    alignas(64) int8_t hidden_q[S4_AMX_ROWS_PER_TASK * S4_D_FF];
    alignas(64) float hidden_scales[S4_AMX_ROWS_PER_TASK];
    alignas(64)
        int32_t down_acc[S4_AMX_ROWS_PER_TASK * S4_D_MODEL];

    s3_amx_projection(input, task.padded_rows, S4_D_MODEL, gate,
                      S4_D_FF, gate_acc);
    s3_amx_projection(input, task.padded_rows, S4_D_MODEL, up,
                      S4_D_FF, up_acc);

    for (int row = 0; row < task.valid_rows; row++) {
        const __m512 gate_scale =
            _mm512_set1_ps(input_scales[row] * gate_weight_scale);
        const __m512 up_scale =
            _mm512_set1_ps(input_scales[row] * up_weight_scale);
        for (int f = 0; f < S4_D_FF; f += 16) {
            const size_t offset = (size_t)row * S4_D_FF + f;
            const __m512 gate_value = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    (const __m512i*)(gate_acc + offset))),
                gate_scale);
            const __m512 up_value = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    (const __m512i*)(up_acc + offset))),
                up_scale);
            const __m512 sigmoid = reciprocal512_ps(_mm512_add_ps(
                _mm512_set1_ps(1.0f),
                exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(),
                                       gate_value))));
            _mm512_store_ps(
                hidden + offset,
                _mm512_mul_ps(
                    _mm512_mul_ps(gate_value, sigmoid), up_value));
        }
        hidden_scales[row] = quantize_s3_s8(
            hidden + (size_t)row * S4_D_FF, S4_D_FF,
            hidden_q + (size_t)row * S4_D_FF);
    }
    if (task.padded_rows > task.valid_rows) {
        std::memset(
            hidden_q + (size_t)task.valid_rows * S4_D_FF, 0,
            (size_t)(task.padded_rows - task.valid_rows) * S4_D_FF);
    }

    s3_amx_projection(hidden_q, task.padded_rows, S4_D_FF, down,
                      S4_D_MODEL, down_acc);
    for (int row = 0; row < task.valid_rows; row++) {
        const int source_row = task.first_row + row;
        const int token =
            shared ? source_row : work.grouped_tokens[source_row];
        const float mixture =
            shared ? 1.0f : work.grouped_mixtures[source_row];
        const __m512 scale = _mm512_set1_ps(
            hidden_scales[row] * down_weight_scale * mixture);
        for (int d = 0; d < S4_D_MODEL; d += 16) {
            const __m512 result = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    (const __m512i*)(down_acc +
                                     (size_t)row * S4_D_MODEL + d))),
                scale);
            if (shared) {
                _mm512_storeu_ps(
                    work.y + (size_t)token * S4_D_MODEL + d,
                    _mm512_add_ps(
                        _mm512_loadu_ps(
                            work.x + (size_t)token * S4_D_MODEL + d),
                        result));
            } else {
                const int rank = work.grouped_ranks[source_row];
                _mm512_store_ps(
                    work.routed_output[rank][token] + d, result);
            }
        }
    }
}

static void s4_build_tasks(S4WorkContext& work) {
    work.task_count = 0;
    for (int first = 0; first < S4_NUM_TOKENS;
         first += S4_SHARED_ROWS_PER_TASK) {
        const int valid =
            std::min(S4_SHARED_ROWS_PER_TASK, S4_NUM_TOKENS - first);
        const int padded =
            (valid + S3_TILE_ROWS - 1) & -S3_TILE_ROWS;
        S4Task& task = work.tasks[work.task_count++];
        task = {S4TaskKind::SharedAmx, -1, first, valid, padded,
                2800 + 900 * (padded / S3_TILE_ROWS)};
    }

    for (int expert = 0; expert < S4_NUM_EXPERTS; expert++) {
        const int rows = work.expert_counts[expert];
        if (rows == 0) continue;
        if (rows < S4_AMX_THRESHOLD) {
            S4Task& task = work.tasks[work.task_count++];
            task = {S4TaskKind::RoutedVnni, expert,
                    work.expert_offsets[expert], rows, rows,
                    3400 * rows + 500};
            continue;
        }

        int consumed = 0;
        while (consumed < rows) {
            const int valid =
                std::min(S4_AMX_ROWS_PER_TASK, rows - consumed);
            const int padded =
                (valid + S3_TILE_ROWS - 1) & -S3_TILE_ROWS;
            S4Task& task = work.tasks[work.task_count++];
            task = {S4TaskKind::RoutedAmx, expert,
                    work.expert_offsets[expert] + consumed, valid, padded,
                    2800 + 900 * (padded / S3_TILE_ROWS)};
            consumed += valid;
        }
    }

    int task_order[S4_MAX_TASKS];
    for (int task = 0; task < work.task_count; task++) {
        task_order[task] = task;
    }
    std::sort(task_order, task_order + work.task_count,
              [&](int first, int second) {
                  return work.tasks[first].cost > work.tasks[second].cost;
              });
    int thread_cost[S4_THREADS] = {};
    std::memset(work.thread_task_count, 0,
                sizeof(work.thread_task_count));
    for (int position = 0; position < work.task_count; position++) {
        const int task = task_order[position];
        int target = 0;
        const int eligible_threads =
            work.tasks[task].kind == S4TaskKind::RoutedVnni
                ? S4_THREADS
                : S4_THREADS / 2;
        for (int thread = 1; thread < eligible_threads; thread++) {
            if (thread_cost[thread] < thread_cost[target]) {
                target = thread;
            }
        }
        work.thread_tasks[target][work.thread_task_count[target]++] =
            task;
        thread_cost[target] += work.tasks[task].cost;
    }
}

static void s4_execute_worker(int thread, S4WorkContext& work) {
    const MoEWeights& w = *work.weights;
    const int first_token =
        thread * (S4_NUM_TOKENS / S4_THREADS);
    const int last_token =
        first_token + S4_NUM_TOKENS / S4_THREADS;
    int* local_counts = work.local_counts[thread];
    std::memset(local_counts, 0,
                S4_NUM_EXPERTS * sizeof(local_counts[0]));
    static thread_local bool amx_ready = request_s3_amx_permission();
    if (!amx_ready) std::abort();

    for (int token = first_token; token < last_token; token++) {
        work.input_scales[token] = quantize_s3_s8(
            work.x + (size_t)token * S4_D_MODEL, S4_D_MODEL,
            work.xq[token]);
    }
    s4_barrier(work.quantize_arrived);

    if (thread < S4_THREADS / 2) {
        constexpr int router_rows =
            S4_NUM_TOKENS / (S4_THREADS / 2);
        const int router_first = thread * router_rows;
        _tile_loadconfig(&s3_tile_config);
        s3_amx_projection(
            work.xq[router_first], router_rows, S4_D_MODEL,
            s4_router_packed, S4_NUM_EXPERTS,
            &work.router_acc[router_first][0]);
        _tile_release();
    }
    s4_barrier(work.router_arrived);

    for (int token = first_token; token < last_token; token++) {
        s4_router_select(
            work.x + (size_t)token * S4_D_MODEL,
            work.router_acc[token], work.input_scales[token], w,
            work.selected_experts[token],
            work.selected_mixtures[token]);
        local_counts[work.selected_experts[token][0]]++;
        local_counts[work.selected_experts[token][1]]++;
    }

    s4_barrier(work.route_arrived);
    if (thread == 0) {
        int grouped_row = 0;
        for (int expert = 0; expert < S4_NUM_EXPERTS; expert++) {
            work.expert_offsets[expert] = grouped_row;
            int count = 0;
            for (int owner = 0; owner < S4_THREADS; owner++) {
                work.thread_offsets[owner][expert] =
                    grouped_row + count;
                count += work.local_counts[owner][expert];
            }
            work.expert_counts[expert] = count;
            const int padded =
                (count + S3_TILE_ROWS - 1) & -S3_TILE_ROWS;
            if (padded > count) {
                std::memset(work.grouped_xq[grouped_row + count], 0,
                            (size_t)(padded - count) * S4_D_MODEL);
            }
            grouped_row += padded;
        }
        work.expert_offsets[S4_NUM_EXPERTS] = grouped_row;
        s4_build_tasks(work);
        work.offsets_ready.store(1, std::memory_order_release);
    } else {
        while (work.offsets_ready.load(std::memory_order_acquire) == 0) {
            _mm_pause();
        }
    }

    int cursor[S4_NUM_EXPERTS];
    std::memcpy(cursor, work.thread_offsets[thread], sizeof(cursor));
    for (int token = first_token; token < last_token; token++) {
        for (int rank = 0; rank < S4_TOP_K; rank++) {
            const int expert = work.selected_experts[token][rank];
            const int row = cursor[expert]++;
            std::memcpy(work.grouped_xq[row], work.xq[token],
                        S4_D_MODEL);
            work.grouped_scales[row] = work.input_scales[token];
            work.grouped_tokens[row] = token;
            work.grouped_ranks[row] = rank;
            work.grouped_mixtures[row] =
                work.selected_mixtures[token][rank];
        }
    }
    s4_barrier(work.scatter_arrived);

    _tile_loadconfig(&s3_tile_config);
    for (int index = 0; index < work.thread_task_count[thread];
         index++) {
        const S4Task& task =
            work.tasks[work.thread_tasks[thread][index]];
        if (task.kind == S4TaskKind::RoutedVnni) {
            for (int row = task.first_row;
                 row < task.first_row + task.valid_rows; row++) {
                s4_vnni_expert(work, task.expert, row);
            }
        } else {
            s4_amx_expert(work, task);
        }
    }
    _tile_release();

    s4_barrier(work.expert_arrived);
    for (int token = first_token; token < last_token; token++) {
        for (int d = 0; d < S4_D_MODEL; d += 16) {
            __m512 output = _mm512_loadu_ps(
                work.y + (size_t)token * S4_D_MODEL + d);
            output = _mm512_add_ps(
                output,
                _mm512_load_ps(work.routed_output[0][token] + d));
            output = _mm512_add_ps(
                output,
                _mm512_load_ps(work.routed_output[1][token] + d));
            _mm512_storeu_ps(
                work.y + (size_t)token * S4_D_MODEL + d, output);
        }
    }
}

class S4WorkerPool {
   public:
    ~S4WorkerPool() {
        stop.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (int i = 0; i < S4_THREADS - 1; i++) {
            if (workers[i].joinable()) workers[i].join();
        }
    }

    void start() {
        if (started) return;
        started = true;
        for (int i = 0; i < S4_THREADS - 1; i++) {
            workers[i] =
                std::thread(&S4WorkerPool::worker_loop, this, i + 1);
        }
        while (ready.load(std::memory_order_acquire) !=
               S4_THREADS - 1) {
            _mm_pause();
        }
#ifdef _OPENMP
        pin_workers();
#endif
    }

    void run(S4WorkContext& work) {
        context.store(&work, std::memory_order_relaxed);
        const uint64_t current = ++next_generation;
        generation.store(current, std::memory_order_release);
        s4_execute_worker(0, work);
        for (int i = 0; i < S4_THREADS - 1; i++) {
            while (completed[i].load(std::memory_order_acquire) !=
                   current) {
                _mm_pause();
            }
        }
    }

   private:
#ifdef _OPENMP
    static void pin_one(pthread_t handle, int cpu) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) return;
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        pthread_setaffinity_np(handle, sizeof(affinity), &affinity);
    }

    void pin_workers() {
        int place_cpus[CPU_SETSIZE][2];
        int place_sizes[CPU_SETSIZE] = {};
        const int place_count =
            std::min(omp_get_num_places(), CPU_SETSIZE);
        for (int place = 0; place < place_count; place++) {
            const int count = omp_get_place_num_procs(place);
            if (count <= 0 || count > 2) return;
            int ids[2];
            omp_get_place_proc_ids(place, ids);
            place_sizes[place] = count;
            for (int i = 0; i < count; i++) {
                place_cpus[place][i] = ids[i];
            }
        }

        int cpus[S4_THREADS];
        int cpu_count = 0;
        for (int lane = 0; lane < 2 && cpu_count < S4_THREADS; lane++) {
            for (int place = 0;
                 place < place_count && cpu_count < S4_THREADS; place++) {
                if (lane < place_sizes[place]) {
                    cpus[cpu_count++] = place_cpus[place][lane];
                }
            }
        }
        if (cpu_count < S4_THREADS) return;
        pin_one(pthread_self(), cpus[0]);
        for (int i = 0; i < S4_THREADS - 1; i++) {
            pin_one(workers[i].native_handle(), cpus[i + 1]);
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
            S4WorkContext* work =
                context.load(std::memory_order_relaxed);
            s4_execute_worker(thread, *work);
            completed[thread - 1].store(current,
                                         std::memory_order_release);
        }
    }

    std::thread workers[S4_THREADS - 1];
    alignas(64) std::atomic<uint64_t> completed[S4_THREADS - 1];
    alignas(64) std::atomic<S4WorkContext*> context{nullptr};
    alignas(64) std::atomic<uint64_t> generation{0};
    alignas(64) std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    uint64_t next_generation = 0;
    bool started = false;
};

static S4WorkContext* s4_context = nullptr;

static void initialize_s4_context() {
    if (!s4_context) s4_context = new S4WorkContext();
}
static S4WorkerPool& s4_worker_pool() {
    static S4WorkerPool pool;
    return pool;
}

static void start_s4_worker_pool() { s4_worker_pool().start(); }

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
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && \
    defined(__AMX_TILE__) && defined(__AMX_INT8__)
    if (!s4_preprocessed || !s4_context || num_tokens != S4_NUM_TOKENS) {
        moe_forward_generic(x, w, y, num_tokens);
        return;
    }

    S4WorkContext& work = *s4_context;
    work.x = x;
    work.weights = &w;
    work.y = y;
    work.route_arrived.store(0, std::memory_order_relaxed);
    work.quantize_arrived.store(0, std::memory_order_relaxed);
    work.router_arrived.store(0, std::memory_order_relaxed);
    work.offsets_ready.store(0, std::memory_order_relaxed);
    work.scatter_arrived.store(0, std::memory_order_relaxed);
    work.expert_arrived.store(0, std::memory_order_relaxed);
    s4_worker_pool().run(work);
#else
    moe_forward_generic(x, w, y, num_tokens);
#endif
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

#endif  // defined(__riscv)
