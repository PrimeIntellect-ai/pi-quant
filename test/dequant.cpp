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
constexpr double epsilon {1e-1};

using namespace piquant;

#define test_dequant(ti, to, rnd, reduce) \
    TEST(dequantize, dequantize_##ti##_to_##to##_##rnd##_##reduce) { \
        std::random_device rd {}; \
        std::mt19937 gen {rd()}; \
        std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)}; \
            std::size_t numel_out {is_int4<to> ? (numel+1)>>1 : numel}; \
            \
            std::vector<ti> data_in {}; \
            std::vector<to> quantized {}; \
            data_in.resize(numel); \
            quantized.resize(numel_out); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            auto [scale, zero_point] {piquant::compute_quant_config_from_data(data_in, std::numeric_limits<std::make_signed_t<to>>::max())}; \
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            ctx.reseed_thread_local_rng(9'3'2002); \
            ctx.quantize_generic<ti, to>(data_in, quantized, scale, zero_point, piquant::round_mode::rnd); \
            std::vector<ti> dequantized {}; \
            dequantized.resize(numel); \
            ti prev {piquant::reduce_op::reduce == piquant::reduce_op::add ? dist(gen) : 0.0f}; \
            std::ranges::fill(dequantized, prev); \
            ctx.dequantize_generic<to, ti>(quantized, dequantized, scale, zero_point, piquant::reduce_op::reduce); \
            for (std::size_t i {}; i < numel; ++i) { \
                ASSERT_NEAR(data_in[i], dequantized[i]-prev, epsilon); \
            } \
        } \
    }

//test_dequant(float, uint4_t, nearest, set)
//test_dequant(float, uint4_t, stochastic, set)
//test_dequant(float, uint4_t, nearest, add) TODO
//test_dequant(float, uint4_t, stochastic, add)
test_dequant(float, uint8_t, nearest, set)
test_dequant(float, uint8_t, stochastic, set)
test_dequant(float, uint8_t, nearest, add)
test_dequant(float, uint8_t, stochastic, add)
test_dequant(float, uint16_t, nearest, set)
test_dequant(float, uint16_t, stochastic, set)
test_dequant(float, uint16_t, nearest, add)
test_dequant(float, uint16_t, stochastic, add)
test_dequant(float, uint32_t, nearest, set)
test_dequant(float, uint32_t, stochastic, set)
test_dequant(float, uint32_t, nearest, add)
test_dequant(float, uint32_t, stochastic, add)
//test_dequant(float, uint64_t, nearest, set) TODO
//test_dequant(float, uint64_t, stochastic, set)
//test_dequant(float, uint64_t, nearest, add)
//test_dequant(float, uint64_t, stochastic, add)
//test_dequant(float, int4_t, nearest, set)
//test_dequant(float, int4_t, stochastic, set)
//test_dequant(float, int4_t, nearest, add) TODO
//test_dequant(float, int4_t, stochastic, add)
test_dequant(float, int8_t, nearest, set)
test_dequant(float, int8_t, stochastic, set)
test_dequant(float, int8_t, nearest, add)
test_dequant(float, int8_t, stochastic, add)
test_dequant(float, int16_t, nearest, set)
test_dequant(float, int16_t, stochastic, set)
test_dequant(float, int16_t, nearest, add)
test_dequant(float, int16_t, stochastic, add)
test_dequant(float, int32_t, nearest, set)
test_dequant(float, int32_t, stochastic, set)
test_dequant(float, int32_t, nearest, add)
test_dequant(float, int32_t, stochastic, add)
test_dequant(float, int64_t, nearest, set)
test_dequant(float, int64_t, stochastic, set)
test_dequant(float, int64_t, nearest, add)
test_dequant(float, int64_t, stochastic, add)
//test_dequant(double, uint4_t, nearest, set) TODO
//test_dequant(double, uint4_t, stochastic, set)
//test_dequant(double, uint4_t, nearest, add)
//test_dequant(double, uint4_t, stochastic, add)
test_dequant(double, uint8_t, nearest, set)
test_dequant(double, uint8_t, stochastic, set)
test_dequant(double, uint8_t, nearest, add)
test_dequant(double, uint8_t, stochastic, add)
test_dequant(double, uint16_t, nearest, set)
test_dequant(double, uint16_t, stochastic, set)
test_dequant(double, uint16_t, nearest, add)
test_dequant(double, uint16_t, stochastic, add)
test_dequant(double, uint32_t, nearest, set)
test_dequant(double, uint32_t, stochastic, set)
test_dequant(double, uint32_t, nearest, add)
test_dequant(double, uint32_t, stochastic, add)
//test_dequant(double, uint64_t, nearest, set) TODO
//test_dequant(double, uint64_t, stochastic, set)
//test_dequant(double, uint64_t, nearest, add)
//test_dequant(double, uint64_t, stochastic, add)
//test_dequant(double, int4_t, nearest, set) TODO
//test_dequant(double, int4_t, stochastic, set)
//test_dequant(double, int4_t, nearest, add)
//test_dequant(double, int4_t, stochastic, add)
test_dequant(double, int8_t, nearest, set)
test_dequant(double, int8_t, stochastic, set)
test_dequant(double, int8_t, nearest, add)
test_dequant(double, int8_t, stochastic, add)
test_dequant(double, int16_t, nearest, set)
test_dequant(double, int16_t, stochastic, set)
test_dequant(double, int16_t, nearest, add)
test_dequant(double, int16_t, stochastic, add)
test_dequant(double, int32_t, nearest, set)
test_dequant(double, int32_t, stochastic, set)
test_dequant(double, int32_t, nearest, add)
test_dequant(double, int32_t, stochastic, add)
test_dequant(double, int64_t, nearest, set)
test_dequant(double, int64_t, stochastic, set)
test_dequant(double, int64_t, nearest, add)
test_dequant(double, int64_t, stochastic, add)
