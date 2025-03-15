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

constexpr std::size_t iters {1000};

TEST(piquant, uint8_round_nearest) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};

    for (std::size_t n {}; n < iters; ++n) {
        float scale {std::uniform_real_distribution<float>{0.1f, 1.0f}(gen)};
        std::int32_t zero_point {std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)};
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};

        std::vector<piquant::f32> data_in {};
        std::vector<piquant::quint8> data_out_naive {};
        std::vector<piquant::quint8> data_out {};
        data_in.resize(numel);
        data_out.resize(numel);
        data_out_naive.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        q8_naive(data_in, data_out_naive, scale, zero_point);
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<piquant::f32, piquant::quint8>(data_in, data_out, scale, zero_point, piquant::round_mode::nearest);
        for (std::size_t i {}; i < numel; ++i) {
            if (data_out[i] != data_out_naive[i]) {
                std::cout << "Mismatch at index " << i << ": " << static_cast<int>(data_out[i]) << " != " << static_cast<int>(data_out_naive[i]) << std::endl;
                std::cout << "Input: " << data_in[i] << std::endl;
                ASSERT_EQ(static_cast<int>(data_out[i]), static_cast<int>(data_out_naive[i]));
            }
        }
    }
}

TEST(piquant, uint8_round_nearest_025) {
    std::random_device rd {};
    std::mt19937 gen {rd()};

    for (std::size_t n {}; n < iters; ++n) {
        float scale {std::uniform_real_distribution<float>{0.1f, 1.0f}(gen)};
        std::int32_t zero_point {std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)};
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};

        std::vector<piquant::f32> data_in {};
        std::vector<piquant::quint8> data_out_naive {};
        std::vector<piquant::quint8> data_out {};
        data_in.resize(numel);
        data_out.resize(numel);
        data_out_naive.resize(numel);
        std::ranges::generate(data_in, [&] { return 0.25f; });
        q8_naive(data_in, data_out_naive, scale, zero_point);
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<piquant::f32, piquant::quint8>(data_in, data_out, scale, zero_point, piquant::round_mode::nearest);
        for (std::size_t i {}; i < numel; ++i) {
            if (data_out[i] != data_out_naive[i]) {
                std::cout << "Mismatch at index " << i << ": " << static_cast<int>(data_out[i]) << " != " << static_cast<int>(data_out_naive[i]) << std::endl;
                std::cout << "Input: " << data_in[i] << std::endl;
                ASSERT_EQ(static_cast<int>(data_out[i]), static_cast<int>(data_out_naive[i]));
            }
        }
    }
}

TEST(piquant, uint8_round_stochastic) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};

    for (std::size_t n {}; n < iters; ++n) {
        std::vector<std::array<float, 3>> avgs {};
        float scale {std::uniform_real_distribution<float>{0.1f, 1.0f}(gen)};
        std::int32_t zero_point {std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)};
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
        avgs.reserve(iters);
        std::vector<piquant::f32> data_in {};
        std::vector<piquant::quint8> data_out_sto {};
        std::vector<piquant::quint8> data_out_near {};
        data_in.resize(numel);
        data_out_sto.resize(numel);
        data_out_near.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<piquant::f32, piquant::quint8>(data_in, data_out_near, scale, zero_point, piquant::round_mode::nearest);
        ctx.quantize_generic<piquant::f32, piquant::quint8>(data_in, data_out_sto, scale, zero_point, piquant::round_mode::stochastic);
        std::vector<piquant::f32> dequant_near {};
        std::vector<piquant::f32> dequant_sto {};
        dequant_near.resize(numel);
        dequant_sto.resize(numel);
        ctx.dequantize_generic<piquant::quint8, piquant::f32>(data_out_near, dequant_near, scale, zero_point, piquant::reduce_op::set);
        ctx.dequantize_generic<piquant::quint8, piquant::f32>(data_out_sto, dequant_sto, scale, zero_point, piquant::reduce_op::set);
        float avg_near {std::accumulate(dequant_near.begin(), dequant_near.end(), 0.0f) / static_cast<float>(numel)};
        float avg_sto {std::accumulate(dequant_sto.begin(), dequant_sto.end(), 0.0f) / static_cast<float>(numel)};
        float avg_original {std::accumulate(data_in.begin(), data_in.end(), 0.0f) / static_cast<float>(numel)};
        avgs.push_back({avg_near, avg_sto, avg_original});
    }
}

TEST(piquant, uint4_round_nearest) {
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};

    for (std::size_t n {}; n < iters; ++n) {
        float scale {std::uniform_real_distribution<float>{0.1f, 1.0f}(gen)};
        std::int32_t zero_point {std::uniform_int_distribution<std::int32_t>{-128, 127}(gen)};
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
        std::size_t out_numel {(numel + 1)>>1};

        std::vector<piquant::f32> data_in {};
        std::vector<piquant::quint4> data_out_naive {};
        std::vector<piquant::quint4> data_out {};
        data_in.resize(numel);
        data_out.resize(out_numel);
        data_out_naive.resize(out_numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });

        q4_naive(data_in, data_out_naive, scale, zero_point);
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<piquant::f32, piquant::quint4>(data_in, data_out, scale, zero_point, piquant::round_mode::nearest);

        for (std::size_t i {}; i < out_numel; ++i) {
            auto a = data_out_naive[i];
            auto b = data_out[i];
            if (a != b) {
                std::cerr << "Mismatch at index " << i << ": " << static_cast<int>(a) << " != " << static_cast<int>(b) << std::endl;
                std::abort();
            }
        }
    }
}

