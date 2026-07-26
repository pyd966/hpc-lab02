#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace {

constexpr int kMatrixCount = 15;
constexpr int kMatrixBytes = 1024 * 512;
constexpr size_t kWeightBytes = (size_t)kMatrixCount * kMatrixBytes;
constexpr int kAmxBlocks = 6;

struct MatrixSpec {
    int n;
    int k;
    size_t output_offset;
};

struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved0[14];
    uint16_t colsb[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

static_assert(sizeof(TileConfig) == 64);

volatile uint64_t sink;

void* allocate_aligned(size_t bytes) {
    void* memory = nullptr;
    if (posix_memalign(&memory, 64, bytes) != 0) return nullptr;
    return memory;
}

std::vector<MatrixSpec> make_specs(size_t& output_count) {
    std::vector<MatrixSpec> specs;
    output_count = 0;
    for (int expert = 0; expert < 5; expert++) {
        specs.push_back({512, 1024, output_count});
        output_count += 512;
        specs.push_back({512, 1024, output_count});
        output_count += 512;
        specs.push_back({1024, 512, output_count});
        output_count += 1024;
    }
    return specs;
}

void pack_vnni(const int8_t* source, const MatrixSpec& spec, int8_t* packed,
               int32_t* row_sums) {
    for (int row = 0; row < spec.n; row++) {
        int32_t sum = 0;
        for (int col = 0; col < spec.k; col++) {
            sum += source[(size_t)row * spec.k + col];
        }
        row_sums[row] = sum;
    }

    constexpr int tile_blocks = 16;
    const int output_blocks = spec.n / 16;
    const int reduction_blocks = spec.k / 4;
    for (int tile = 0; tile < output_blocks / tile_blocks; tile++) {
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
                        source + (size_t)(ob * 16 + lane) * spec.k + rb * 4;
                    std::memcpy(block + lane * 4, row, 4);
                }
            }
        }
    }
}

void pack_amx(const int8_t* source, const MatrixSpec& spec, int8_t* packed) {
    const int output_blocks = spec.n / 16;
    const int k_chunks = spec.k / 64;
    size_t cursor = 0;
    for (int ob = 0; ob < output_blocks; ob += kAmxBlocks) {
        const int blocks = std::min(kAmxBlocks, output_blocks - ob);
        for (int kc = 0; kc < k_chunks; kc++) {
            for (int block = 0; block < blocks; block++) {
                for (int row = 0; row < 16; row++) {
                    const int8_t* input =
                        source + (size_t)(ob + block) * 16 * spec.k +
                        (size_t)row * spec.k + kc * 64;
                    std::memcpy(packed + cursor, input, 64);
                    cursor += 64;
                }
            }
        }
    }
}

inline __m512i dpbusd_memory(__m512i accumulator, __m512i activation,
                            const int8_t* weights) {
    asm("vpdpbusd %2, %1, %0"
        : "+v"(accumulator)
        : "v"(activation), "m"(*(const __m512i*)weights));
    return accumulator;
}

__attribute__((noinline)) uint64_t run_load_roof(const int8_t* weights) {
    asm volatile("" : : "r"(weights) : "memory");
    __m512i acc[8] = {
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512(),
        _mm512_setzero_si512(), _mm512_setzero_si512()};
    for (size_t offset = 0; offset < kWeightBytes; offset += 8 * 64) {
#pragma GCC unroll 8
        for (int i = 0; i < 8; i++) {
            acc[i] = _mm512_add_epi64(
                acc[i],
                _mm512_load_si512((const __m512i*)(weights + offset + i * 64)));
        }
    }
    __m512i total = acc[0];
    for (int i = 1; i < 8; i++) total = _mm512_add_epi64(total, acc[i]);
    return (uint64_t)_mm512_reduce_add_epi64(total);
}

__attribute__((noinline)) uint64_t run_vnni(
    const std::vector<MatrixSpec>& specs, const int8_t* weights,
    const int32_t* row_sums, const uint8_t* x1024, const uint8_t* x512,
    int32_t* output) {
    constexpr int tile_blocks = 16;
    size_t sums_offset = 0;
    for (int matrix = 0; matrix < kMatrixCount; matrix++) {
        const MatrixSpec& spec = specs[matrix];
        const int reduction_blocks = spec.k / 4;
        const int output_blocks = spec.n / 16;
        const int8_t* matrix_weights = weights + (size_t)matrix * kMatrixBytes;
        const uint8_t* activation_bytes = spec.k == 1024 ? x1024 : x512;

        for (int tile = 0; tile < output_blocks / tile_blocks; tile++) {
            __m512i acc[tile_blocks];
#pragma GCC unroll 16
            for (int ob = 0; ob < tile_blocks; ob++) {
                const int global_ob = tile * tile_blocks + ob;
                const __m512i correction = _mm512_slli_epi32(
                    _mm512_load_si512((const __m512i*)(
                        row_sums + sums_offset + global_ob * 16)),
                    7);
                acc[ob] =
                    _mm512_sub_epi32(_mm512_setzero_si512(), correction);
            }
            const size_t tile_base =
                (size_t)tile * reduction_blocks * tile_blocks * 64;
            for (int rb = 0; rb < reduction_blocks; rb++) {
                uint32_t activation4;
                std::memcpy(&activation4, activation_bytes + rb * 4, 4);
                const __m512i activation =
                    _mm512_set1_epi32((int)activation4);
#pragma GCC unroll 16
                for (int ob = 0; ob < tile_blocks; ob++) {
                    const size_t offset =
                        tile_base + ((size_t)rb * tile_blocks + ob) * 64;
                    acc[ob] = dpbusd_memory(acc[ob], activation,
                                            matrix_weights + offset);
                }
            }
#pragma GCC unroll 16
            for (int ob = 0; ob < tile_blocks; ob++) {
                const int global_ob = tile * tile_blocks + ob;
                _mm512_store_si512(
                    (__m512i*)(output + spec.output_offset + global_ob * 16),
                    acc[ob]);
            }
        }
        sums_offset += spec.n;
    }

    uint64_t checksum = 0;
    for (size_t i = 0; i < specs.back().output_offset + specs.back().n; i++) {
        checksum += (uint32_t)output[i];
    }
    return checksum;
}

void configure_amx(bool partial) {
    TileConfig config{};
    config.palette_id = 1;
    config.rows[0] = 16;
    config.colsb[0] = 64;
    config.rows[1] = 16;
    config.colsb[1] = partial ? 64 : 4;
    for (int tile = 2; tile < 8; tile++) {
        config.rows[tile] = 16;
        config.colsb[tile] = partial ? 64 : 4;
    }
    _tile_loadconfig(&config);
}

inline void zero_accumulator(int block) {
    switch (block) {
        case 0: _tile_zero(2); break;
        case 1: _tile_zero(3); break;
        case 2: _tile_zero(4); break;
        case 3: _tile_zero(5); break;
        case 4: _tile_zero(6); break;
        case 5: _tile_zero(7); break;
    }
}

inline void amx_dot_block(int block, const int8_t* weights) {
    switch (block) {
        case 0:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(2, 0, 1);
            break;
        case 1:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(3, 0, 1);
            break;
        case 2:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(4, 0, 1);
            break;
        case 3:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(5, 0, 1);
            break;
        case 4:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(6, 0, 1);
            break;
        case 5:
            _tile_loadd(0, weights, 64);
            _tile_dpbssd(7, 0, 1);
            break;
    }
}

inline void zero_six_accumulators() {
    _tile_zero(2);
    _tile_zero(3);
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
}

inline void amx_dot_six(const int8_t* weights) {
    _tile_loadd(0, weights, 64);
    _tile_dpbssd(2, 0, 1);
    _tile_loadd(0, weights + 1024, 64);
    _tile_dpbssd(3, 0, 1);
    _tile_loadd(0, weights + 2048, 64);
    _tile_dpbssd(4, 0, 1);
    _tile_loadd(0, weights + 3072, 64);
    _tile_dpbssd(5, 0, 1);
    _tile_loadd(0, weights + 4096, 64);
    _tile_dpbssd(6, 0, 1);
    _tile_loadd(0, weights + 5120, 64);
    _tile_dpbssd(7, 0, 1);
}

inline void store_standard_block(int block, int32_t* output) {
    switch (block) {
        case 0: _tile_stored(2, output, 4); break;
        case 1: _tile_stored(3, output, 4); break;
        case 2: _tile_stored(4, output, 4); break;
        case 3: _tile_stored(5, output, 4); break;
        case 4: _tile_stored(6, output, 4); break;
        case 5: _tile_stored(7, output, 4); break;
    }
}

inline void store_partial_block(int block, int32_t* output) {
    switch (block) {
        case 0: _tile_stored(2, output, 64); break;
        case 1: _tile_stored(3, output, 64); break;
        case 2: _tile_stored(4, output, 64); break;
        case 3: _tile_stored(5, output, 64); break;
        case 4: _tile_stored(6, output, 64); break;
        case 5: _tile_stored(7, output, 64); break;
    }
}

__attribute__((noinline)) uint64_t run_amx_standard(
    const std::vector<MatrixSpec>& specs, const int8_t* weights,
    const int8_t* x1024, const int8_t* x512, int32_t* output) {
    for (int matrix = 0; matrix < kMatrixCount; matrix++) {
        const MatrixSpec& spec = specs[matrix];
        const int output_blocks = spec.n / 16;
        const int k_chunks = spec.k / 64;
        const int8_t* matrix_weights = weights + (size_t)matrix * kMatrixBytes;
        const int8_t* activation = spec.k == 1024 ? x1024 : x512;
        size_t cursor = 0;

        for (int ob = 0; ob < output_blocks; ob += kAmxBlocks) {
            const int blocks = std::min(kAmxBlocks, output_blocks - ob);
            if (blocks == kAmxBlocks) {
                zero_six_accumulators();
                for (int kc = 0; kc < k_chunks; kc++) {
                    _tile_loadd(1, activation + kc * 64, 4);
                    amx_dot_six(matrix_weights + cursor +
                                (size_t)kc * kAmxBlocks * 1024);
                }
            } else {
                for (int block = 0; block < blocks; block++) {
                    zero_accumulator(block);
                }
                for (int kc = 0; kc < k_chunks; kc++) {
                    _tile_loadd(1, activation + kc * 64, 4);
                    for (int block = 0; block < blocks; block++) {
                        const int8_t* tile =
                            matrix_weights + cursor +
                            ((size_t)kc * blocks + block) * 1024;
                        amx_dot_block(block, tile);
                    }
                }
            }
            for (int block = 0; block < blocks; block++) {
                store_standard_block(
                    block, output + spec.output_offset + (ob + block) * 16);
            }
            cursor += (size_t)blocks * k_chunks * 1024;
        }
    }

    uint64_t checksum = 0;
    for (size_t i = 0; i < specs.back().output_offset + specs.back().n; i++) {
        checksum += (uint32_t)output[i];
    }
    return checksum;
}

void build_diagonal_activation(const int8_t* activation, int k,
                               int8_t* diagonal) {
    std::memset(diagonal, 0, (size_t)(k / 64) * 1024);
    for (int kc = 0; kc < k / 64; kc++) {
        for (int group = 0; group < 16; group++) {
            std::memcpy(diagonal + (size_t)kc * 1024 + group * 64 + group * 4,
                        activation + kc * 64 + group * 4, 4);
        }
    }
}

__attribute__((noinline)) uint64_t run_amx_partial(
    const std::vector<MatrixSpec>& specs, const int8_t* weights,
    const int8_t* diagonal1024, const int8_t* diagonal512, int32_t* output) {
    alignas(64) int32_t partial[16][16];
    for (int matrix = 0; matrix < kMatrixCount; matrix++) {
        const MatrixSpec& spec = specs[matrix];
        const int output_blocks = spec.n / 16;
        const int k_chunks = spec.k / 64;
        const int8_t* matrix_weights = weights + (size_t)matrix * kMatrixBytes;
        const int8_t* diagonal =
            spec.k == 1024 ? diagonal1024 : diagonal512;
        size_t cursor = 0;

        for (int ob = 0; ob < output_blocks; ob += kAmxBlocks) {
            const int blocks = std::min(kAmxBlocks, output_blocks - ob);
            if (blocks == kAmxBlocks) {
                zero_six_accumulators();
                for (int kc = 0; kc < k_chunks; kc++) {
                    _tile_loadd(1, diagonal + (size_t)kc * 1024, 64);
                    amx_dot_six(matrix_weights + cursor +
                                (size_t)kc * kAmxBlocks * 1024);
                }
            } else {
                for (int block = 0; block < blocks; block++) {
                    zero_accumulator(block);
                }
                for (int kc = 0; kc < k_chunks; kc++) {
                    _tile_loadd(1, diagonal + (size_t)kc * 1024, 64);
                    for (int block = 0; block < blocks; block++) {
                        const int8_t* tile =
                            matrix_weights + cursor +
                            ((size_t)kc * blocks + block) * 1024;
                        amx_dot_block(block, tile);
                    }
                }
            }
            for (int block = 0; block < blocks; block++) {
                store_partial_block(block, &partial[0][0]);
                for (int row = 0; row < 16; row++) {
                    int32_t sum = 0;
                    for (int col = 0; col < 16; col++) {
                        sum += partial[row][col];
                    }
                    output[spec.output_offset + (ob + block) * 16 + row] = sum;
                }
            }
            cursor += (size_t)blocks * k_chunks * 1024;
        }
    }

    uint64_t checksum = 0;
    for (size_t i = 0; i < specs.back().output_offset + specs.back().n; i++) {
        checksum += (uint32_t)output[i];
    }
    return checksum;
}

template <class Function>
double time_kernel(int iterations, Function&& function) {
    const auto begin = std::chrono::steady_clock::now();
    uint64_t checksum = 0;
    for (int i = 0; i < iterations; i++) checksum += function();
    const auto end = std::chrono::steady_clock::now();
    sink = checksum;
    return std::chrono::duration<double>(end - begin).count();
}

void print_result(const char* name, double seconds, int iterations) {
    const double ns = seconds * 1.0e9 / iterations;
    const double bandwidth =
        (double)kWeightBytes * iterations / seconds / 1.0e9;
    const double useful_gops =
        2.0 * kWeightBytes * iterations / seconds / 1.0e9;
    std::printf("%-18s %9.1f ns/pass  %6.2f GB/s  %7.2f useful GOPS\n",
                name, ns, bandwidth, useful_gops);
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
    if (iterations <= 0) return 1;

    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        std::perror("ARCH_REQ_XCOMP_PERM");
        return 2;
    }

    size_t output_count;
    const std::vector<MatrixSpec> specs = make_specs(output_count);
    int8_t* source = (int8_t*)allocate_aligned(kWeightBytes);
    int8_t* vnni_weights = (int8_t*)allocate_aligned(kWeightBytes);
    int8_t* amx_weights = (int8_t*)allocate_aligned(kWeightBytes);
    int32_t* row_sums =
        (int32_t*)allocate_aligned(output_count * sizeof(int32_t));
    int32_t* vnni_output =
        (int32_t*)allocate_aligned(output_count * sizeof(int32_t));
    int32_t* amx_output =
        (int32_t*)allocate_aligned(output_count * sizeof(int32_t));
    int32_t* partial_output =
        (int32_t*)allocate_aligned(output_count * sizeof(int32_t));
    if (!source || !vnni_weights || !amx_weights || !row_sums ||
        !vnni_output || !amx_output || !partial_output) {
        std::fprintf(stderr, "allocation failed\n");
        return 3;
    }

    alignas(64) int8_t x1024[1024];
    alignas(64) int8_t x512[512];
    alignas(64) uint8_t shifted1024[1024];
    alignas(64) uint8_t shifted512[512];
    alignas(64) int8_t diagonal1024[16 * 1024];
    alignas(64) int8_t diagonal512[8 * 1024];

    for (size_t i = 0; i < kWeightBytes; i++) {
        source[i] = (int8_t)((i * 37 + 11) % 255 - 127);
    }
    for (int i = 0; i < 1024; i++) {
        x1024[i] = (int8_t)((i * 29 + 7) % 255 - 127);
        shifted1024[i] = (uint8_t)((int)x1024[i] + 128);
        if (i < 512) {
            x512[i] = x1024[i];
            shifted512[i] = shifted1024[i];
        }
    }

    size_t sums_offset = 0;
    for (int matrix = 0; matrix < kMatrixCount; matrix++) {
        pack_vnni(source + (size_t)matrix * kMatrixBytes, specs[matrix],
                  vnni_weights + (size_t)matrix * kMatrixBytes,
                  row_sums + sums_offset);
        pack_amx(source + (size_t)matrix * kMatrixBytes, specs[matrix],
                 amx_weights + (size_t)matrix * kMatrixBytes);
        sums_offset += specs[matrix].n;
    }
    build_diagonal_activation(x1024, 1024, diagonal1024);
    build_diagonal_activation(x512, 512, diagonal512);

    const uint64_t vnni_check =
        run_vnni(specs, vnni_weights, row_sums, shifted1024, shifted512,
                 vnni_output);
    configure_amx(false);
    const uint64_t amx_check =
        run_amx_standard(specs, amx_weights, x1024, x512, amx_output);
    _tile_release();
    configure_amx(true);
    const uint64_t partial_check =
        run_amx_partial(specs, amx_weights, diagonal1024, diagonal512,
                        partial_output);
    _tile_release();

    bool correct = vnni_check == amx_check && vnni_check == partial_check;
    for (size_t i = 0; i < output_count; i++) {
        correct &= vnni_output[i] == amx_output[i];
        correct &= vnni_output[i] == partial_output[i];
    }
    std::printf("S2 active weights: %.2f MiB, useful INT8 MACs/pass: %zu\n",
                kWeightBytes / 1048576.0, kWeightBytes);
    std::printf("Correctness: %s (checksums %llu / %llu / %llu)\n",
                correct ? "PASS" : "FAIL",
                (unsigned long long)vnni_check,
                (unsigned long long)amx_check,
                (unsigned long long)partial_check);
    if (!correct) return 4;

    for (int warmup = 0; warmup < 3; warmup++) {
        sink = run_load_roof(vnni_weights);
    }
    const double load_seconds =
        time_kernel(iterations, [&] { return run_load_roof(vnni_weights); });

    for (int warmup = 0; warmup < 3; warmup++) {
        sink = run_vnni(specs, vnni_weights, row_sums, shifted1024,
                        shifted512, vnni_output);
    }
    const double vnni_seconds = time_kernel(iterations, [&] {
        return run_vnni(specs, vnni_weights, row_sums, shifted1024,
                        shifted512, vnni_output);
    });

    configure_amx(false);
    for (int warmup = 0; warmup < 3; warmup++) {
        sink = run_amx_standard(specs, amx_weights, x1024, x512, amx_output);
    }
    const double amx_seconds = time_kernel(iterations, [&] {
        return run_amx_standard(specs, amx_weights, x1024, x512, amx_output);
    });
    _tile_release();

    configure_amx(true);
    for (int warmup = 0; warmup < 3; warmup++) {
        sink = run_amx_partial(specs, amx_weights, diagonal1024, diagonal512,
                               partial_output);
    }
    const double partial_seconds = time_kernel(iterations, [&] {
        return run_amx_partial(specs, amx_weights, diagonal1024, diagonal512,
                               partial_output);
    });
    _tile_release();

    print_result("load roof", load_seconds, iterations);
    print_result("VNNI dot roof", vnni_seconds, iterations);
    print_result("AMX GEMV", amx_seconds, iterations);
    print_result("AMX K-partial", partial_seconds, iterations);

    std::free(source);
    std::free(vnni_weights);
    std::free(amx_weights);
    std::free(row_sums);
    std::free(vnni_output);
    std::free(amx_output);
    std::free(partial_output);
    return 0;
}
