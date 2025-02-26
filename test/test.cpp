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

#include "naive.hpp"

TEST(uint8, round_nearest) {
    std::cout << "uint8 quant Test..." << std::endl;
    constexpr std::size_t numel {1'000'000};
    std::vector<float> data_in {};
    std::vector<std::uint8_t> data_out_naive {};
    std::vector<std::uint8_t> data_out {};
    data_in.resize(numel);
    data_out.resize(numel);
    data_out_naive.resize(numel);
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
    std::ranges::generate(data_in, [&] { return dist(gen); });

    q8_naive(data_in, data_out_naive, 0.5, 0);
    quant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
    ctx.quantize_uint8(data_in, data_out, 0.5, 0, quant::round_mode::nearest);

    for (std::size_t i {}; i < numel; ++i) {
        auto a = data_out_naive[i];
        auto b = data_out[i];
        ASSERT_EQ(a, b);
    }

    std::cout << "uint8 quant Test passed!" << std::endl;
}

TEST(uint8, round_stochastic) {
    std::cout << "uint8 Stochastic Quant Test..." << std::endl;
    constexpr std::size_t numel {1'000'000};
    constexpr std::size_t iters {10};
    constexpr auto zp {128};
    constexpr auto scale {0.00784f};
    std::vector<std::array<float, 3>> avgs {};
    avgs.reserve(iters);
    for (std::size_t i=0; i < iters; ++i) {
        std::vector<float> data_in {};
        std::vector<std::uint8_t> data_out_sto {};
        std::vector<std::uint8_t> data_out_near {};
        data_in.resize(numel);
        data_out_sto.resize(numel);
        data_out_near.resize(numel);
        std::random_device rd {};
        std::mt19937 gen {rd()};
        std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
        std::ranges::generate(data_in, [&] { return dist(gen); });
        quant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_uint8(data_in, data_out_near, scale, zp, quant::round_mode::nearest);
        ctx.quantize_uint8(data_in, data_out_sto, scale, zp, quant::round_mode::stochastic);
        std::vector<float> dequant_near {};
        std::vector<float> dequant_sto {};
        dequant_near.resize(numel);
        dequant_sto.resize(numel);
        ctx.dequantize_uint8(data_out_near, dequant_near, scale, zp);
        ctx.dequantize_uint8(data_out_sto, dequant_sto, scale, zp);
        float avg_near {std::accumulate(dequant_near.begin(), dequant_near.end(), 0.0f) / static_cast<float>(numel)};
        float avg_sto {std::accumulate(dequant_sto.begin(), dequant_sto.end(), 0.0f) / static_cast<float>(numel)};
        float avg_original {std::accumulate(data_in.begin(), data_in.end(), 0.0f) / static_cast<float>(numel)};
        avgs.push_back({avg_near, avg_sto, avg_original});
    }

    float avg_near {std::accumulate(avgs.begin(), avgs.end(), 0.0f, [](float acc, const auto& v) { return acc + v[0]; }) / static_cast<float>(iters)};
    float avg_sto {std::accumulate(avgs.begin(), avgs.end(), 0.0f, [](float acc, const auto& v) { return acc + v[1]; }) / static_cast<float>(iters)};
    float avg_original {std::accumulate(avgs.begin(), avgs.end(), 0.0f, [](float acc, const auto& v) { return acc + v[2]; }) / static_cast<float>(iters)};

    std::cout << std::fixed << std::showpoint;
    std::cout << "Original avg: " << avg_original << std::endl;
    std::cout << "Quantized uint8 Nearest avg: " << avg_near << std::endl;
    std::cout << "Quantized uint8 Stochastic avg: " << avg_sto << std::endl;

    std::cout << "uint8 stochastic Test passed!" << std::endl;
}

TEST(uint4, round_nearest) {
    std::cout << "uint4 quant Test" << std::endl;
    constexpr std::size_t numel {32};
    constexpr std::size_t out_numel {(numel + 1)/2};
    constexpr auto zp {128};
    constexpr auto scale {0.00784f};
    std::vector<float> data_in {};
    std::vector<std::uint8_t> data_out_naive {};
    std::vector<std::uint8_t> data_out {};
    data_in.resize(numel);
    data_out.resize(out_numel);
    data_out_naive.resize(out_numel);
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
    std::ranges::generate(data_in, [&] { return dist(gen); });

    q4_naive(data_in, data_out_naive, scale, zp);
    quant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
    ctx.quantize_uint4(data_in, data_out, scale, zp, quant::round_mode::nearest);

    for (std::size_t i {}; i < out_numel; ++i) {
        auto a = data_out_naive[i];
        auto b = data_out[i];
        if (a != b) {
            std::cerr << "Mismatch at index " << i << ": " << static_cast<int>(a) << " != " << static_cast<int>(b) << std::endl;
            std::abort();
        }
    }

    std::cout << "uint4 quant Test passed!" << std::endl;
}

