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
constexpr std::int32_t stochastic_epsilon {1};

using namespace piquant;

template <const std::uint8_t IDX, typename T>
[[nodiscard]] constexpr auto unpack_nibble(T val, bool is_signed) noexcept -> std::int32_t {
    const uint8_t raw = (static_cast<std::uint8_t>(val.bits) >> (IDX<<2)) & 0xF;
    if (!is_signed) {
        return raw;
    }
    return (raw & 0x8) ? static_cast<std::int32_t>(raw) - 16 : static_cast<std::int32_t>(raw);
}

#define test_quant(ti, to, rnd) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<fp32_t> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            fp32_t scale {std::uniform_real_distribution<fp32_t>{0.1, 1.0}(gen)}; \
            std::int32_t zero_point {std::is_same_v<uint4_t, to> ? std::uniform_int_distribution<std::int32_t>{-8, 7}(gen) : \
                    std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)}; \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)}; \
            std::size_t numel_out {std::is_same_v<uint4_t, to> ? (numel+1)>>1 : numel}; \
            \
            std::vector<ti> data_in {}; \
            std::vector<to> data_out_naive {}; \
            std::vector<to> data_out {}; \
            data_in.resize(numel); \
            data_out.resize(numel_out); \
            data_out_naive.resize(numel_out); \
            std::ranges::generate(data_in, [&] { return dist(gen); }); \
            quantize_naive<ti, to, piquant::round_mode::rnd>(data_in, data_out_naive, scale, zero_point); \
            piquant::context ctx {std::max(1u, 4u)}; \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                    eq |= std::abs(static_cast<std::int32_t>(data_out[i]) - static_cast<std::int32_t>(data_out_naive[i])) <= stochastic_epsilon; \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << static_cast<std::int32_t>(data_out[i]) << " != " << static_cast<std::int32_t>(data_out_naive[i]) << std::endl; \
                    std::cout << "Input: " << static_cast<fp32_t>(data_in[i]) << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

#define test_quant_int4(ti, to, rnd, is_stochastic, is_signed) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<fp32_t> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            std::cout << "Iteration " << n << std::endl; \
            fp32_t scale {std::uniform_real_distribution<fp32_t>{0.1, 1.0}(gen)}; \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)}; \
            std::size_t numel_out {std::is_same_v<uint4_t, to> ? (numel+1)>>1 : numel}; \
            std::int32_t zero_point {std::is_same_v<uint4_t, to> ? std::uniform_int_distribution<std::int32_t>{-8, 7}(gen) : \
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
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                std::int32_t a {unpack_nibble<0>(data_out[i], is_signed)}; \
                std::int32_t b {unpack_nibble<1>(data_out[i], is_signed)}; \
                std::int32_t a_naive {unpack_nibble<0>(data_out_naive[i], is_signed)}; \
                std::int32_t b_naive {unpack_nibble<1>(data_out_naive[i], is_signed)}; \
                if (is_stochastic) { \
                    eq |= std::abs(a - a_naive) <= stochastic_epsilon; \
                    eq |= std::abs(b - b_naive) <= stochastic_epsilon; \
                } else { \
                    eq = eq && (a == a_naive) && (b == b_naive); \
                } \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << "(" << a << ", " << b << ") != (" << a_naive << ", " << b_naive << ") -> " << (std::int32_t)(data_out[i].bits) << " != " << (std::int32_t)(data_out_naive[i].bits) << std::endl; \
                    std::cout << "Input: " << static_cast<fp32_t>(data_in[i]) << std::endl; \
                    std::cout << "Data in: ["; \
                    for (std::size_t j {}; j < numel; ++j) { \
                        std::cout << static_cast<fp32_t>(data_in[j]) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (f): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<std::int32_t>(data_out[j].bits) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (n): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<std::int32_t>(data_out_naive[j].bits) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Num el out: " << numel_out << std::endl; \
                    std::cout << "Num el in: " << numel << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

#define test_quant_int2(ti, to, rnd, is_stochastic, is_signed) \
    TEST(quantize, quantize_##ti##_to_##to##_##rnd) { \
        std::mt19937 gen {0x9032002}; \
        std::uniform_real_distribution<fp32_t> dist {-1.0, 1.0}; \
        \
        for (std::size_t n {}; n < iters; ++n) { \
            std::cout << "Iteration " << n << std::endl; \
            fp32_t scale {std::uniform_real_distribution<fp32_t>{0.1, 1.0}(gen)}; \
            std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)}; \
            std::size_t numel_out {std::is_same_v<uint2_t, to> ? (numel+3)>>2 : numel}; \
            std::int32_t zero_point {std::is_same_v<uint2_t, to> ? std::uniform_int_distribution<std::int32_t>{-1, 2}(gen) : \
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
            piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())}; \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                std::int32_t a {unpack_nibble<0>(data_out[i], is_signed)}; \
                std::int32_t b {unpack_nibble<1>(data_out[i], is_signed)}; \
                std::int32_t a_naive {unpack_nibble<0>(data_out_naive[i], is_signed)}; \
                std::int32_t b_naive {unpack_nibble<1>(data_out_naive[i], is_signed)}; \
                if (is_stochastic) { \
                    eq |= std::abs(a - a_naive) <= stochastic_epsilon; \
                    eq |= std::abs(b - b_naive) <= stochastic_epsilon; \
                } else { \
                    eq = eq && (a == a_naive) && (b == b_naive); \
                } \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << "(" << a << ", " << b << ") != (" << a_naive << ", " << b_naive << ") -> " << (std::int32_t)(data_out[i].bits) << " != " << (std::int32_t)(data_out_naive[i].bits) << std::endl; \
                    std::cout << "Input: " << static_cast<fp32_t>(data_in[i]) << std::endl; \
                    std::cout << "Data in: ["; \
                    for (std::size_t j {}; j < numel; ++j) { \
                        std::cout << static_cast<fp32_t>(data_in[j]) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (f): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<std::int32_t>(data_out[j].bits) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Data out (n): ["; \
                    for (std::size_t j {}; j < numel_out; ++j) { \
                        std::cout << static_cast<std::int32_t>(data_out_naive[j].bits) << " "; \
                    } \
                    std::cout << "]" << std::endl; \
                    std::cout << "Num el out: " << numel_out << std::endl; \
                    std::cout << "Num el in: " << numel << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

test_quant_int4(fp32_t, uint4_t, nearest, false, false)
test_quant_int4(fp32_t, uint4_t, stochastic, true, false)
test_quant_int4(bfp16_t, uint4_t, nearest, false, false)
test_quant_int4(bfp16_t, uint4_t, stochastic, true, false)
test_quant(fp32_t, uint8_t, nearest)
test_quant(fp32_t, uint8_t, stochastic)
test_quant(bfp16_t, uint8_t, nearest)
test_quant(bfp16_t, uint8_t, stochastic)

TEST(quantize, requantize_float_to_uint8_identity_data) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::size_t numel {std::uniform_int_distribution<std::size_t>{5000, 1'5000}(gen)};
    std::size_t numel_out {numel};
    std::vector<fp32_t> data_in {};
    std::vector<std::uint8_t> quantized {};
    data_in.resize(numel);
    quantized.resize(numel_out);
    std::ranges::fill(data_in, 42.0f);
    context ctx {std::max(1u, std::thread::hardware_concurrency())};
    auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<std::uint8_t>::type_code)};
    ctx.quantize_generic<fp32_t, std::uint8_t>(data_in, quantized, scale, zero_point, round_mode::nearest);
    std::vector<fp32_t> dequantized {};
    dequantized.resize(numel);
    ctx.dequantize_generic<std::uint8_t, fp32_t>(quantized, dequantized, scale, zero_point, reduce_op::add);
    for (std::size_t i {}; i < numel; ++i) {
        ASSERT_NEAR(data_in[i], dequantized[i], 1e-6f);
    }
}
