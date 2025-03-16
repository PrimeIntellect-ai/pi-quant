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
            std::size_t numel_out {is_int4<to> ? numel+1>>1 : numel}; \
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
            ctx.reseed_thread_local_rng(9'3'2002); \
            ctx.quantize_generic<ti, to>(data_in, data_out, scale, zero_point, piquant::round_mode::rnd); \
            for (std::size_t i {}; i < numel_out; ++i) { \
                bool eq {eq = data_out[i] == data_out_naive[i]}; \
                if (piquant::round_mode::rnd == piquant::round_mode::stochastic) { \
                    if (is_int4<to>) { \
                        eq |= std::abs(static_cast<int>(((int)data_out[i]>>4)&0x0F) - static_cast<int>(((int)data_out_naive[i]>>4)&0x0F)) <= stochastic_epsilon \
                            && std::abs(static_cast<int>((int)data_out[i+1]&0x0F) - static_cast<int>((int)data_out_naive[i+1]&0x0F)) <= stochastic_epsilon; \
                    } else { \
                        eq |= std::abs(static_cast<int>(data_out[i]) - static_cast<int>(data_out_naive[i])) <= stochastic_epsilon; \
                    } \
                } \
                if (!eq) { \
                    std::cout << "Mismatch at index " << i << ": " << static_cast<int>(data_out[i]) << " != " << static_cast<int>(data_out_naive[i]) << std::endl; \
                    std::cout << "Input: " << data_in[i] << std::endl; \
                    ASSERT_TRUE(eq); \
                } \
            } \
        } \
    }

test_quant(float, uint4_t, nearest)
test_quant(float, uint4_t, stochastic)
test_quant(float, uint8_t, nearest)
test_quant(float, uint8_t, stochastic)
test_quant(float, uint16_t, nearest)
test_quant(float, uint16_t, stochastic)
test_quant(float, uint32_t, nearest)
test_quant(float, uint32_t, stochastic)
test_quant(float, uint64_t, nearest)
test_quant(float, uint64_t, stochastic)
//test_quant(float, int4_t, nearest)
//test_quant(float, int4_t, stochastic)  TODO: same seed
test_quant(float, int8_t, nearest)
test_quant(float, int8_t, stochastic)
test_quant(float, int16_t, nearest)
test_quant(float, int16_t, stochastic)
test_quant(float, int32_t, nearest)
test_quant(float, int32_t, stochastic)
test_quant(float, int64_t, nearest)
test_quant(float, int64_t, stochastic)
test_quant(double, uint4_t, nearest)
test_quant(double, uint4_t, stochastic)
test_quant(double, uint8_t, nearest)
test_quant(double, uint8_t, stochastic)
test_quant(double, uint16_t, nearest)
test_quant(double, uint16_t, stochastic)
test_quant(double, uint32_t, nearest)
test_quant(double, uint32_t, stochastic)
test_quant(double, uint64_t, nearest)
test_quant(double, uint64_t, stochastic)
//test_quant(double, int4_t, nearest)  TODO: same seed
//test_quant(double, int4_t, stochastic)
test_quant(double, int8_t, nearest)
test_quant(double, int8_t, stochastic)
test_quant(double, int16_t, nearest)
test_quant(double, int16_t, stochastic)
test_quant(double, int32_t, nearest)
test_quant(double, int32_t, stochastic)
test_quant(double, int64_t, nearest)
test_quant(double, int64_t, stochastic)

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
        std::vector<float> data_in {};
        std::vector<std::uint8_t> data_out_sto {};
        std::vector<std::uint8_t> data_out_near {};
        data_in.resize(numel);
        data_out_sto.resize(numel);
        data_out_near.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<float, std::uint8_t>(data_in, data_out_near, scale, zero_point, piquant::round_mode::nearest);
        ctx.quantize_generic<float, std::uint8_t>(data_in, data_out_sto, scale, zero_point, piquant::round_mode::stochastic);
        std::vector<float> dequant_near {};
        std::vector<float> dequant_sto {};
        dequant_near.resize(numel);
        dequant_sto.resize(numel);
        ctx.dequantize_generic<std::uint8_t, float>(data_out_near, dequant_near, scale, zero_point, piquant::reduce_op::set);
        ctx.dequantize_generic<std::uint8_t, float>(data_out_sto, dequant_sto, scale, zero_point, piquant::reduce_op::set);
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

        std::vector<float> data_in {};
        std::vector<piquant::uint4_t> data_out_naive {};
        std::vector<piquant::uint4_t> data_out {};
        data_in.resize(numel);
        data_out.resize(out_numel);
        data_out_naive.resize(out_numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });

        quantize_naive_4bit(data_in, data_out_naive, scale, zero_point);
        piquant::context ctx {std::max(1u, std::thread::hardware_concurrency())};
        ctx.quantize_generic<float, piquant::uint4_t>(data_in, data_out, scale, zero_point, piquant::round_mode::nearest);

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

