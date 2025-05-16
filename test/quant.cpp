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
constexpr int stochastic_epsilon {1};

using namespace piquant;

template <const std::uint8_t IDX, typename T>
[[nodiscard]] constexpr auto unpack_nibble(T val, bool is_signed) noexcept -> int {
    const uint8_t raw = (static_cast<std::uint8_t>(val.u8) >> (IDX<<2)) & 0xF;
    if (!is_signed) {
        return raw;
    }
    return (raw & 0x8) ? static_cast<int>(raw) - 16 : static_cast<int>(raw);
}

#define test_quant(ti, to, rnd) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            ti scale {std::uniform_real_distribution<ti>{0.1, 1.0}(gen)}; \
            std::int32_t zero_point {is_int4<to> ? std::uniform_int_distribution<std::int32_t>{-8, 7}(gen) : \
                    std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)}; \
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
                    eq |= std::abs(static_cast<int>(data_out[i]) - static_cast<int>(data_out_naive[i])) <= stochastic_epsilon; \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << static_cast<int>(data_out[i]) << " != " << static_cast<int>(data_out_naive[i]) << std::endl; \
                    std::cout << "Input: " << data_in[i] << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

#define test_quant_int4(ti, to, rnd, is_stochastic, is_signed) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<ti> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            std::cout << "Iteration " << n << std::endl; \
            ti scale {std::uniform_real_distribution<ti>{0.1, 1.0}(gen)}; \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)}; \
            std::size_t numel_out {is_int4<to> ? (numel+1)>>1 : numel}; \
            std::int32_t zero_point {is_int4<to> ? std::uniform_int_distribution<std::int32_t>{-8, 7}(gen) : \
                    std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)}; \
            \
            std::vector<ti> data_in {}; \
            std::vector<to> data_out_naive {}; \
            std::vector<to> data_out {}; \
            data_in.resize(numel); \
            data_out.resize(numel_out); \
            data_out_naive.resize(numel_out); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            quantize_naive<ti, to, piquant::round_mode::rnd>(data_in, data_out_naive, scale, zero_point); \
            piquant::context ctx {1}; \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                int a {unpack_nibble<0>(data_out[i], is_signed)}; \
                int b {unpack_nibble<1>(data_out[i], is_signed)}; \
                int a_naive {unpack_nibble<0>(data_out_naive[i], is_signed)}; \
                int b_naive {unpack_nibble<1>(data_out_naive[i], is_signed)}; \
                if (is_stochastic) { \
                    eq |= std::abs(a - a_naive) <= stochastic_epsilon; \
                    eq |= std::abs(b - b_naive) <= stochastic_epsilon; \
                } else { \
                    eq = eq && (a == a_naive) && (b == b_naive); \
                } \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << "(" << a << ", " << b << ") != (" << a_naive << ", " << b_naive << ") -> " << (int)static_cast<uint8_t>(data_out[i].u8) << " != " << (int)static_cast<uint8_t>(data_out_naive[i].u8) << std::endl; \
                    std::cout << "Input: " << data_in[i] << std::endl; \
                    std::cout << "Data in: ["; \
                    for (std::size_t j {}; j < numel; ++j) { \
                        std::cout << data_in[j] << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (f): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<int>(data_out[j].u8) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (n): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<int>(data_out_naive[j].u8) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Num el out: " << numel_out << std::endl; \
                    std::cout << "Num el in: " << numel << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

test_quant_int4(float, uint4_t, nearest, false, false)
test_quant_int4(float, uint4_t, stochastic, true, false)
test_quant(float, uint8_t, nearest)
test_quant(float, uint8_t, stochastic)
test_quant(float, uint16_t, nearest)
test_quant(float, uint16_t, stochastic)
test_quant(float, uint32_t, nearest)
test_quant(float, uint32_t, stochastic)
test_quant(float, uint64_t, nearest)
test_quant(float, uint64_t, stochastic)
test_quant_int4(float, int4_t, nearest, false, true)
test_quant_int4(float, int4_t, stochastic, true, true)
test_quant(float, int8_t, nearest)
test_quant(float, int8_t, stochastic)
test_quant(float, int16_t, nearest)
test_quant(float, int16_t, stochastic)
test_quant(float, int32_t, nearest)
test_quant(float, int32_t, stochastic)
test_quant(float, int64_t, nearest)
test_quant(float, int64_t, stochastic)
test_quant_int4(double, uint4_t, nearest, false, false)
test_quant_int4(double, uint4_t, stochastic, true, false)
test_quant(double, uint8_t, nearest)
test_quant(double, uint8_t, stochastic)
test_quant(double, uint16_t, nearest)
test_quant(double, uint16_t, stochastic)
test_quant(double, uint32_t, nearest)
test_quant(double, uint32_t, stochastic)
test_quant(double, uint64_t, nearest)
test_quant(double, uint64_t, stochastic)
test_quant_int4(double, int4_t, nearest, false, true)
test_quant_int4(double, int4_t, stochastic, true, true)
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
    std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
    std::size_t numel_out {numel};
    std::vector<float> data_in {};
    std::vector<std::uint8_t> quantized {};
    data_in.resize(numel);
    quantized.resize(numel_out);
    std::ranges::fill(data_in, 42.0f);
    context ctx {std::max(1u, std::thread::hardware_concurrency())};
    auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<std::uint8_t>::type_code)};
    ctx.quantize_generic<float, std::uint8_t>(data_in, quantized, scale, zero_point, round_mode::nearest);
    std::vector<float> dequantized {};
    dequantized.resize(numel);
    ctx.dequantize_generic<std::uint8_t, float>(quantized, dequantized, scale, zero_point, reduce_op::add);
    for (std::size_t i {}; i < numel; ++i) {
        ASSERT_NEAR(data_in[i], dequantized[i], 1e-6f);
    }
}

TEST(quantize, requantize_double_to_uint8_identity_data) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
    std::size_t numel_out {numel};
    std::vector<double> data_in {};
    std::vector<std::uint8_t> quantized {};
    data_in.resize(numel);
    quantized.resize(numel_out);
    std::ranges::fill(data_in, 42.0);
    context ctx {std::max(1u, std::thread::hardware_concurrency())};
    auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<std::uint8_t>::type_code)};
    ctx.quantize_generic<double, std::uint8_t>(data_in, quantized, scale, zero_point, round_mode::nearest);
    std::vector<double> dequantized {};
    dequantized.resize(numel);
    ctx.dequantize_generic<std::uint8_t, double>(quantized, dequantized, scale, zero_point, reduce_op::add);
    for (std::size_t i {}; i < numel; ++i) {
        ASSERT_NEAR(data_in[i], dequantized[i], 1e-6);
    }
}
