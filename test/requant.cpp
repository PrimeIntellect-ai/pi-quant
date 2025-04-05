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

constexpr std::size_t iters {100};
constexpr double epsilon {1e-1};

using namespace piquant;

#define test_requant(ti, to, rnd, reduce) \
    TEST(requantize, requantize_##ti##_to_##to##_##rnd##_##reduce) { \
    std::random_device rd {}; \
    std::mt19937 gen {rd()}; \
    std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
    \
    for (std::size_t n {}; n < iters; ++n) { \
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)}; \
        \
        std::vector<ti> data_in {}; \
        data_in.resize(numel); \
        std::ranges::generate(data_in, [&] { return dist(gen); }); \
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
        auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<to>::ty)}; \
        std::vector<ti> requantized {}; \
        requantized.resize(numel); \
        ti prev {piquant::reduce_op::reduce == piquant::reduce_op::add ? dist(gen) : 0.0f}; \
        std::ranges::fill(requantized, prev); \
        ctx.quantize_dequantize_fused_generic<ti, to>(data_in, requantized, scale, zero_point, piquant::round_mode::rnd, piquant::reduce_op::reduce); \
        for (std::size_t i {}; i < numel; ++i) { \
            ASSERT_NEAR(data_in[i], requantized[i]-prev, epsilon); \
        } \
        } \
    }

//test_requant(float, uint4_t, nearest, set) TODO
//test_requant(float, uint4_t, stochastic, set)
//test_requant(float, uint4_t, nearest, add)
//test_requant(float, uint4_t, stochastic, add)
test_requant(float, uint8_t, nearest, set)
test_requant(float, uint8_t, stochastic, set)
test_requant(float, uint8_t, nearest, add)
test_requant(float, uint8_t, stochastic, add)
test_requant(float, uint16_t, nearest, set)
test_requant(float, uint16_t, stochastic, set)
test_requant(float, uint16_t, nearest, add)
test_requant(float, uint16_t, stochastic, add)
test_requant(float, uint32_t, nearest, set)
test_requant(float, uint32_t, stochastic, set)
test_requant(float, uint32_t, nearest, add)
test_requant(float, uint32_t, stochastic, add)
//test_requant(float, uint64_t, nearest, set)
//test_requant(float, uint64_t, stochastic, set)
//test_requant(float, uint64_t, nearest, add)
//test_requant(float, uint64_t, stochastic, add)
//test_requant(float, int4_t, nearest, set)
//test_requant(float, int4_t, stochastic, set)
//test_requant(float, int4_t, nearest, add)
//test_requant(float, int4_t, stochastic, add)
test_requant(float, int8_t, nearest, set)
test_requant(float, int8_t, stochastic, set)
test_requant(float, int8_t, nearest, add)
test_requant(float, int8_t, stochastic, add)
test_requant(float, int16_t, nearest, set)
test_requant(float, int16_t, stochastic, set)
test_requant(float, int16_t, nearest, add)
test_requant(float, int16_t, stochastic, add)
test_requant(float, int32_t, nearest, set)
test_requant(float, int32_t, stochastic, set)
test_requant(float, int32_t, nearest, add)
test_requant(float, int32_t, stochastic, add)
test_requant(float, int64_t, nearest, set)
test_requant(float, int64_t, stochastic, set)
test_requant(float, int64_t, nearest, add)
test_requant(float, int64_t, stochastic, add)
//test_requant(double, uint4_t, nearest, set)
//test_requant(double, uint4_t, stochastic, set)
//test_requant(double, uint4_t, nearest, add)
//test_requant(double, uint4_t, stochastic, add)
test_requant(double, uint8_t, nearest, set)
test_requant(double, uint8_t, stochastic, set)
test_requant(double, uint8_t, nearest, add)
test_requant(double, uint8_t, stochastic, add)
test_requant(double, uint16_t, nearest, set)
test_requant(double, uint16_t, stochastic, set)
test_requant(double, uint16_t, nearest, add)
test_requant(double, uint16_t, stochastic, add)
test_requant(double, uint32_t, nearest, set)
test_requant(double, uint32_t, stochastic, set)
test_requant(double, uint32_t, nearest, add)
test_requant(double, uint32_t, stochastic, add)
//*
// test_requant(double, uint64_t, nearest, set)
//*
// test_requant(double, uint64_t, stochastic, set)
//*
// test_requant(double, uint64_t, nearest, add)
//*
// test_requant(double, uint64_t, stochastic, add)
//test_requant(double, int4_t, nearest, set)
//test_requant(double, int4_t, stochastic, set)
//test_requant(double, int4_t, nearest, add)
//test_requant(double, int4_t, stochastic, add)
test_requant(double, int8_t, nearest, set)
test_requant(double, int8_t, stochastic, set)
test_requant(double, int8_t, nearest, add)
test_requant(double, int8_t, stochastic, add)
test_requant(double, int16_t, nearest, set)
test_requant(double, int16_t, stochastic, set)
test_requant(double, int16_t, nearest, add)
test_requant(double, int16_t, stochastic, add)
test_requant(double, int32_t, nearest, set)
test_requant(double, int32_t, stochastic, set)
test_requant(double, int32_t, nearest, add)
test_requant(double, int32_t, stochastic, add)
test_requant(double, int64_t, nearest, set)
test_requant(double, int64_t, stochastic, set)
test_requant(double, int64_t, nearest, add)
test_requant(double, int64_t, stochastic, add)
