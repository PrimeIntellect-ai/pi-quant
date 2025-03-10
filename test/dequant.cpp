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

constexpr std::size_t iters {1000};

TEST(dequant, uint8_round_nearest_set) {

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
        auto [scale, zero_point] {piquant::compute_quant_config_from_data(data_in)};
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, piquant::round_mode::nearest);

        std::vector<float> dequantized {};
        dequantized.resize(numel);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point, piquant::reduce_op::set);

        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i], 1e-1);
        }
    }
}

TEST(dequant, uint8_round_stochastic_set) {
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
        auto [scale, zero_point] {piquant::compute_quant_config_from_data(data_in)};
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, piquant::round_mode::stochastic);

        std::vector<float> dequantized {};
        dequantized.resize(numel);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point, piquant::reduce_op::set);

        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i], 1e-1);
        }
    }
}

TEST(dequant, uint8_round_nearest_add) {

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
        auto [scale, zero_point] {piquant::compute_quant_config_from_data(data_in)};
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, piquant::round_mode::nearest);

        std::vector<float> dequantized {};
        dequantized.resize(numel);
        float prev {dist(gen)};
        std::ranges::fill(dequantized, prev);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point, piquant::reduce_op::add);
        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i]-prev, 1e-1);
        }
    }
}

TEST(dequant, uint8_round_stochastic_add) {
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
        auto [scale, zero_point] {piquant::compute_quant_config_from_data(data_in)};
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, quantized, scale, zero_point, piquant::round_mode::stochastic);
        std::vector<float> dequantized {};
        dequantized.resize(numel);
        float prev {dist(gen)};
        std::ranges::fill(dequantized, prev);
        ctx.dequantize_uint8(quantized, dequantized, scale, zero_point, piquant::reduce_op::add);
        for (std::size_t i {}; i < numel; ++i) {
            ASSERT_NEAR(data_in[i], dequantized[i]-prev, 1e-1);
        }
    }
}
