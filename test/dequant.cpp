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
constexpr double epsilon {1e-1};

using namespace piquant;

TEST(dequantize, uint4_packing) {
    context ctx {16};

    std::vector<float> input {-1.0f, 1.0f, 2.0f, 3.0f, 0.5f};

    auto [scale, zp] {ctx.compute_quant_config_from_data(input, dtype::uint4)};
    std::cout << "scale: " << scale << " zp: " << +zp << std::endl;

    std::vector<uint4_t> quantized {};
    quantized.resize((input.size()+1)/2);
    ctx.quantize_generic<float, uint4_t>(input, quantized, scale, zp, round_mode::nearest);

    std::vector<float> dequantized {};
    dequantized.resize(input.size());
    ctx.dequantize_generic<uint4_t, float>(quantized, dequantized, scale, zp, reduce_op::set);

    std::cout << "INPUT"  << std::endl;
    for (auto&& x : input)
        std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "OUTPUT"  << std::endl;
    for (auto&& x : dequantized)
        std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "PACKED"  << std::endl;
    for (auto&& x : quantized)
        std::cout << std::bitset<8>(x.u8) << " ";
    std::cout << std::endl;
    std::cout << "UNPACKED"  << std::endl;
    for (auto&& x : quantized)
        std::cout << +x.unpack()[0] << "|" << +x.unpack()[1] << " ";
    std::cout << std::endl;
}

#define test_dequant(ti, to, rnd, reduce) \
    TEST(dequantize, dequantize_##ti##_to_##to##_##rnd##_##reduce) { \
        std::mt19937 gen {0x9032002}; \
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
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<to>::type_code)}; \
            ctx.quantize_generic<ti, to>(data_in, quantized, scale, zero_point, piquant::round_mode::rnd); \
            std::vector<ti> dequantized {}; \
            dequantized.resize(numel); \
            ti prev {piquant::reduce_op::reduce == piquant::reduce_op::add ? dist(gen) : 0.0f}; \
            std::ranges::fill(dequantized, prev); \
            ctx.dequantize_generic<to, ti>(quantized, dequantized, scale, zero_point, piquant::reduce_op::reduce); \
            for (std::size_t i {}; i < numel; ++i) { \
                ASSERT_NEAR(data_in[i], dequantized[i]-prev, epsilon) << "Failed at index " << i << " for n=" << n; \
            } \
        } \
    }

test_dequant(float, uint4_t, nearest, set)
test_dequant(float, uint4_t, stochastic, set)
test_dequant(float, uint4_t, nearest, add)
test_dequant(float, uint4_t, stochastic, add)
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
test_dequant(float, uint64_t, nearest, set)
test_dequant(float, uint64_t, stochastic, set)
test_dequant(float, uint64_t, nearest, add)
test_dequant(float, uint64_t, stochastic, add)
test_dequant(float, int4_t, nearest, set)
test_dequant(float, int4_t, stochastic, set)
test_dequant(float, int4_t, nearest, add)
test_dequant(float, int4_t, stochastic, add)
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
test_dequant(double, uint4_t, nearest, set)
test_dequant(double, uint4_t, stochastic, set)
test_dequant(double, uint4_t, nearest, add)
test_dequant(double, uint4_t, stochastic, add)
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
test_dequant(double, uint64_t, nearest, set)
test_dequant(double, uint64_t, stochastic, set)
test_dequant(double, uint64_t, nearest, add)
test_dequant(double, uint64_t, stochastic, add)
test_dequant(double, int4_t, nearest, set)
test_dequant(double, int4_t, stochastic, set)
test_dequant(double, int4_t, nearest, add)
test_dequant(double, int4_t, stochastic, add)
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
