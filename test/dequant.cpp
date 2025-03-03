#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include <quant.hpp>
#include <gtest/gtest.h>

constexpr std::size_t iters {1000};

TEST(dequant, uint8_round_nearest) {

    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};

    for (std::size_t n {}; n < iters; ++n) {
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};

        std::vector<float> data_in {};
        std::vector<std::uint8_t> quantized {};
        data_in.resize(numel);
        quantized.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        auto [scale, zero_point] {quant::compute_quant_config_from_data(data_in)};
        quant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, quant::round_mode::nearest);

        std::vector<float> dequantized {};
        dequantized.resize(numel);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point);

        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i], 1e-1);
        }
    }
}

TEST(dequant, uint8_round_stochastic) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};

    for (std::size_t n {}; n < iters; ++n) {
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};

        std::vector<float> data_in {};
        std::vector<std::uint8_t> quantized {};
        data_in.resize(numel);
        quantized.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        auto [scale, zero_point] {quant::compute_quant_config_from_data(data_in)};
        quant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, quant::round_mode::stochastic);

        std::vector<float> dequantized {};
        dequantized.resize(numel);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point);

        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i], 1e-1);
        }
    }
}
