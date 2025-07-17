#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>
#include <bitset>

#include <piquant.hpp>
#include <gtest/gtest.h>

constexpr std::size_t iters {10};

using namespace piquant;

#define test_dequant(ti, to, rnd, reduce) \
    TEST(dequantize, dequantize_##ti##_to_##to##_##rnd##_##reduce) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<fp32_t> dist {-1.0, 1.0}; \
        const auto adjusted_epsilon {std::is_same_v<uint2_t, to> ? 2.0f : std::is_same_v<uint4_t, to> ? 0.2f : 0.05}; \
        for (std::size_t n {}; n < iters; ++n) { \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)}; \
            std::size_t numel_out {std::is_same_v<uint2_t, to> ? (numel+3)>>2 : std::is_same_v<uint4_t, to> ? (numel+1)>>1 : numel}; \
            \
            std::vector<ti> data_in {}; \
            std::vector<to> quantized {}; \
            data_in.resize(numel); \
            quantized.resize(numel_out); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            piquant::context ctx {std::max(1u, 4u)}; \
            auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<to>::type_code)}; \
            ctx.quantize_generic<ti, to>(data_in, quantized, scale, zero_point, piquant::round_mode::rnd); \
            std::vector<ti> dequantized {}; \
            dequantized.resize(numel); \
            ti prev {piquant::reduce_op::reduce == piquant::reduce_op::add ? dist(gen) : 0.0f}; \
            std::ranges::fill(dequantized, prev); \
            ctx.dequantize_generic<to, ti>(quantized, dequantized, scale, zero_point, piquant::reduce_op::reduce); \
            for (std::size_t i {}; i < numel; ++i) { \
                const auto a {static_cast<fp32_t>(data_in[i])}; \
                const auto b {static_cast<fp32_t>(dequantized[i]-prev)}; \
                const auto delta {std::abs(a - b)}; \
                bool is_near {delta <= adjusted_epsilon}; \
                if (!is_near) { \
                    std::cout << "Mismatch at index " << i << ": " << a << " != " << b << std::endl; \
                    std::cout << "Numel in: " << numel << " Numel out: " << numel_out << std::endl; \
                    std::cout << "Delta: " << delta << " ZP: " << zero_point << " Scale: " << scale << std::endl; \
                    std::cout << "Zero point: " << zero_point << " Scale: " << scale << std::endl; \
                    std::cout << "IN: ["; \
                    for (std::size_t j {}; j < numel; ++j) { \
                        std::cout << static_cast<fp32_t>(data_in[j]) << ", "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "OT: ["; \
                    for (std::size_t j {}; j < numel; ++j) { \
                        std::cout << static_cast<fp32_t>(dequantized[j]) << ", "; \
                    } \
                    std::cout << "]" << std::endl; \
                    ASSERT_TRUE(is_near); \
                } \
            } \
        } \
    }

test_dequant(fp32_t, uint2_t, nearest, set)
test_dequant(fp32_t, uint2_t, stochastic, set)
test_dequant(fp32_t, uint2_t, nearest, add)
test_dequant(fp32_t, uint2_t, stochastic, add)
test_dequant(fp32_t, uint4_t, nearest, set)
test_dequant(fp32_t, uint4_t, stochastic, set)
test_dequant(fp32_t, uint4_t, nearest, add)
test_dequant(fp32_t, uint4_t, stochastic, add)
test_dequant(fp32_t, uint8_t, nearest, set)
test_dequant(fp32_t, uint8_t, stochastic, set)
test_dequant(fp32_t, uint8_t, nearest, add)
test_dequant(fp32_t, uint8_t, stochastic, add)
test_dequant(bfp16_t, uint2_t, nearest, set)
test_dequant(bfp16_t, uint2_t, stochastic, set)
test_dequant(bfp16_t, uint2_t, nearest, add)
test_dequant(bfp16_t, uint2_t, stochastic, add)
test_dequant(bfp16_t, uint4_t, nearest, set)
test_dequant(bfp16_t, uint4_t, stochastic, set)
test_dequant(bfp16_t, uint4_t, nearest, add)
test_dequant(bfp16_t, uint4_t, stochastic, add)
test_dequant(bfp16_t, uint8_t, nearest, set)
test_dequant(bfp16_t, uint8_t, stochastic, set)
test_dequant(bfp16_t, uint8_t, nearest, add)
test_dequant(bfp16_t, uint8_t, stochastic, add)
