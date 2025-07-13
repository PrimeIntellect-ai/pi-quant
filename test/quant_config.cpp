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

constexpr std::size_t iters {100};
constexpr int stochastic_epsilon {3};

using namespace piquant;

#define test_quant_range(ti, to, rnd) \
    TEST(quantize_range, quantize_range_##ti##_to_##to##_##rnd) { \
        std::random_device rd {}; \
        std::mt19937 gen {rd()}; \
        std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)}; \
            std::size_t numel_out {is_int2<to> ? (numel+3)>>2 : is_int4<to> ? (numel+1)>>1 : numel}; \
            \
            std::vector<to> data_out {}; \
            data_out.resize(numel_out); \
            std::vector<ti> data_in {}; \
            data_in.resize(numel); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<to>::type_code)}; \
            ASSERT_GT(scale, 0.0f); \
            ASSERT_TRUE(std::isfinite(scale)); \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
        } \
    }

test_quant_range(float, int2_t, nearest)
test_quant_range(float, int2_t, stochastic)
test_quant_range(float, uint2_t, nearest)
test_quant_range(float, uint2_t, stochastic)
test_quant_range(float, int4_t, nearest)
test_quant_range(float, int4_t, stochastic)
test_quant_range(float, uint4_t, nearest)
test_quant_range(float, uint4_t, stochastic)
test_quant_range(float, uint8_t, nearest)
test_quant_range(float, uint8_t, stochastic)
test_quant_range(float, uint16_t, nearest)
test_quant_range(float, uint16_t, stochastic)
test_quant_range(float, uint32_t, nearest)
test_quant_range(float, uint32_t, stochastic)
test_quant_range(float, uint64_t, nearest)
test_quant_range(float, uint64_t, stochastic)
test_quant_range(float, int8_t, nearest)
test_quant_range(float, int8_t, stochastic)
test_quant_range(float, int16_t, nearest)
test_quant_range(float, int16_t, stochastic)
test_quant_range(float, int32_t, nearest)
test_quant_range(float, int32_t, stochastic)
test_quant_range(float, int64_t, nearest)
test_quant_range(float, int64_t, stochastic)
test_quant_range(double, uint4_t, nearest)
test_quant_range(double, uint4_t, stochastic)
test_quant_range(double, uint8_t, nearest)
test_quant_range(double, uint8_t, stochastic)
test_quant_range(double, uint16_t, nearest)
test_quant_range(double, uint16_t, stochastic)
test_quant_range(double, uint32_t, nearest)
test_quant_range(double, uint32_t, stochastic)
test_quant_range(double, uint64_t, nearest)
test_quant_range(double, uint64_t, stochastic)
test_quant_range(double, uint2_t, nearest)
test_quant_range(double, uint2_t, stochastic)
test_quant_range(double, int2_t, nearest)
test_quant_range(double, int2_t, stochastic)
test_quant_range(double, int4_t, nearest)
test_quant_range(double, int4_t, stochastic)
test_quant_range(double, int8_t, nearest)
test_quant_range(double, int8_t, stochastic)
test_quant_range(double, int16_t, nearest)
test_quant_range(double, int16_t, stochastic)
test_quant_range(double, int32_t, nearest)
test_quant_range(double, int32_t, stochastic)
test_quant_range(double, int64_t, nearest)
test_quant_range(double, int64_t, stochastic)
