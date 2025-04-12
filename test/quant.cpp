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
constexpr int stochastic_epsilon {3};

using namespace piquant;

#define test_quant(ti, to, rnd) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::random_device rd {}; \
        std::mt19937 gen {rd()}; \
        std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            ti scale {std::uniform_real_distribution<ti>{0.1, 1.0}(gen)}; \
            std::int32_t zero_point {std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)}; \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)}; \
            std::size_t numel_out {is_int4<to> ? (numel+1)>>1 : numel}; \
            \
            std::vector<ti> data_in {}; \
            std::vector<to> data_out_naive {}; \
            std::vector<to> data_out {}; \
            data_in.resize(numel); \
            data_out.resize(numel_out); \
            data_out_naive.resize(numel_out); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            quantize_naive<ti, to, piquant::round_mode::rnd>(data_in, data_out_naive, scale, zero_point); \
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                if (is_int4<to>) { \
                    eq |= std::abs(static_cast<int>(((int)data_out[i]>>4)&0x0F) - static_cast<int>(((int)data_out_naive[i]>>4)&0x0F)) <= stochastic_epsilon \
                        && std::abs(static_cast<int>((int)data_out[i+1]&0x0F) - static_cast<int>((int)data_out_naive[i+1]&0x0F)) <= stochastic_epsilon; \
                } else { \
                    eq |= std::abs(static_cast<int>(data_out[i]) - static_cast<int>(data_out_naive[i])) <= stochastic_epsilon; \
                } \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << static_cast<int>(data_out[i]) << " != " << static_cast<int>(data_out_naive[i]) << std::endl; \
                    std::cout << "Input: " << data_in[i] << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

//test_quant(float, uint4_t, nearest)
//test_quant(float, uint4_t, stochastic)
test_quant(float, uint8_t, nearest)
test_quant(float, uint8_t, stochastic)
test_quant(float, uint16_t, nearest)
test_quant(float, uint16_t, stochastic)
test_quant(float, uint32_t, nearest)
test_quant(float, uint32_t, stochastic)
test_quant(float, uint64_t, nearest)
test_quant(float, uint64_t, stochastic)
//test_quant(float, int4_t, nearest)
//test_quant(float, int4_t, stochastic)
test_quant(float, int8_t, nearest)
test_quant(float, int8_t, stochastic)
test_quant(float, int16_t, nearest)
test_quant(float, int16_t, stochastic)
test_quant(float, int32_t, nearest)
test_quant(float, int32_t, stochastic)
test_quant(float, int64_t, nearest)
test_quant(float, int64_t, stochastic)
//test_quant(double, uint4_t, nearest)
//test_quant(double, uint4_t, stochastic)
test_quant(double, uint8_t, nearest)
test_quant(double, uint8_t, stochastic)
test_quant(double, uint16_t, nearest)
test_quant(double, uint16_t, stochastic)
test_quant(double, uint32_t, nearest)
test_quant(double, uint32_t, stochastic)
test_quant(double, uint64_t, nearest)
test_quant(double, uint64_t, stochastic)
//test_quant(double, int4_t, nearest)
//test_quant(double, int4_t, stochastic)
test_quant(double, int8_t, nearest)
test_quant(double, int8_t, stochastic)
test_quant(double, int16_t, nearest)
test_quant(double, int16_t, stochastic)
test_quant(double, int32_t, nearest)
test_quant(double, int32_t, stochastic)
test_quant(double, int64_t, nearest)
test_quant(double, int64_t, stochastic)

TEST(quantize, requantize_float_to_uint8_identity_data) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0, 1.0};
   
    for (std::size_t n {}; n < iters; ++n) {
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
        std::size_t numel_out {numel};
       
        std::vector<float> data_in {};
        std::vector<std::uint8_t> quantized {};
        data_in.resize(numel);
        quantized.resize(numel_out);
        std::ranges::fill(data_in, dist(gen));
        context ctx {std::max(1u, std::thread::hardware_concurrency())};
        auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<std::uint8_t>::ty)};
        ctx.quantize_generic<float, std::uint8_t>(data_in, quantized, scale, zero_point, round_mode::nearest);
        std::vector<float> dequantized {};
        dequantized.resize(numel);
        ctx.dequantize_generic<std::uint8_t, float>(quantized, dequantized, scale, zero_point, reduce_op::set);
        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i], 1e-4f);
        }
    } 
}
