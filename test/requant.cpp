#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include <piquant.hpp>
#include <gtest/gtest.h>

#include "naive.hpp"

constexpr std::size_t iters {10};
constexpr double epsilon {1e-1};

using namespace piquant;

#define test_requant(ti, to, rnd, reduce) \
    TEST(requantize, requantize_##ti##_to_##to##_##rnd##_##reduce) { \
    std::random_device rd {}; \
    std::mt19937 gen {rd()}; \
    std::uniform_real_distribution<fp32_t> dist {-1.0, 1.0}; \
    const auto adjusted_epsilon {std::is_same_v<uint4_t, to> ? epsilon * 4: epsilon}; \
    for (std::size_t n {}; n < iters; ++n) { \
        std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)}; \
        \
        std::vector<ti> data_in {}; \
        data_in.resize(numel); \
        std::ranges::generate(data_in, [&] { return dist(gen); }); \
        piquant::context ctx {std::max(1u, 4u)}; \
        auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<to>::type_code)}; \
        std::vector<ti> requantized {}; \
        requantized.resize(numel); \
        ti prev {piquant::reduce_op::reduce == piquant::reduce_op::add ? dist(gen) : 0.0f}; \
        std::ranges::fill(requantized, prev); \
        ctx.quantize_dequantize_fused_generic<ti, to>(data_in, requantized, scale, zero_point, piquant::round_mode::rnd, piquant::reduce_op::reduce); \
        for (std::size_t i {}; i < numel; ++i) { \
            ASSERT_NEAR(static_cast<fp32_t>(data_in[i]), static_cast<fp32_t>(requantized[i]-prev), adjusted_epsilon); \
        } \
        } \
    }

test_requant(fp32_t, uint4_t, nearest, set)
test_requant(fp32_t, uint4_t, stochastic, set)
test_requant(fp32_t, uint4_t, nearest, add)
test_requant(fp32_t, uint4_t, stochastic, add)
test_requant(fp32_t, uint8_t, nearest, set)
test_requant(fp32_t, uint8_t, stochastic, set)
test_requant(fp32_t, uint8_t, nearest, add)
test_requant(fp32_t, uint8_t, stochastic, add)
test_requant(bfp16_t, uint4_t, nearest, set)
test_requant(bfp16_t, uint4_t, stochastic, set)
test_requant(bfp16_t, uint4_t, nearest, add)
test_requant(bfp16_t, uint4_t, stochastic, add)
test_requant(bfp16_t, uint8_t, nearest, set)
test_requant(bfp16_t, uint8_t, stochastic, set)
test_requant(bfp16_t, uint8_t, nearest, add)
test_requant(bfp16_t, uint8_t, stochastic, add)
