#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include <quant.hpp>

#include "naive.hpp"

static auto test_q8(std::size_t nt) -> void {
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
    quant::context ctx {nt};
    ctx.quantize_uint8(data_in, data_out, 0.5, 0, quant::round_mode::nearest);

    for (std::size_t i {}; i < numel; ++i) {
        auto a = data_out_naive[i];
        auto b = data_out[i];
        if (a != b) {
            std::cerr << "Mismatch at index " << i << ": " << static_cast<int>(a) << " != " << static_cast<int>(b) << std::endl;
            std::cerr << "Original: " << data_in[i] << std::endl;
            std::abort();
        }
    }

    std::cout << "uint8 quant Test passed!" << std::endl;
}

static auto test_q8_stochastic(std::size_t nt) -> void {
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
        quant::context ctx {nt};
        ctx.quantize_uint8(data_in, data_out_near, scale, zp, quant::round_mode::nearest);
        ctx.quantize_uint8(data_in, data_out_sto, scale, zp, quant::round_mode::stochastic);
        float avg_near {std::accumulate(data_out_near.begin(), data_out_near.end(), 0.0f, [](float acc, std::uint8_t xi) {
            return acc + static_cast<float>(xi);
        }) / static_cast<float>(numel)};
        float avg_sto {std::accumulate(data_out_sto.begin(), data_out_sto.end(), 0.0f, [](float acc, std::uint8_t xi) {
            return acc + static_cast<float>(xi);
        }) / static_cast<float>(numel)};
        float avg_original {std::accumulate(data_in.begin(), data_in.end(), 0.0f, [](float acc, float xi) {
            return acc + xi;
        }) / static_cast<float>(numel)};
        float dequant_avg_near = (avg_near - zp)*scale;
        float dequant_avg_sto  = (avg_sto  - zp)*scale;
        avgs.push_back({dequant_avg_near, dequant_avg_sto, avg_original});
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

static auto test_q4(std::size_t nt) -> void {
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
    quant::context ctx {nt};
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

auto main() -> int {
    const std::size_t nt {std::thread::hardware_concurrency()};
    std::cout << "Num threads: " << nt << std::endl;
    test_q8(nt);
    test_q8_stochastic(nt);
    test_q4(nt);
    return 0;
}

