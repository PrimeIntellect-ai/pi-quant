#include <random>

#include <piquant.hpp>
#include <gtest/gtest.h>

#include "naive.hpp"

constexpr std::size_t iters {1000};

using namespace piquant;

TEST(quant_range, f32) {
    context ctx {std::max(1u, std::thread::hardware_concurrency())};
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
    for (std::size_t n {}; n < iters; ++n) {
        std::size_t numel {std::uniform_int_distribution<std::size_t>{500, 1'500}(gen)};
        std::vector<float> data_in {};
        data_in.resize(numel);
        std::ranges::generate(data_in, [&] { return dist(gen); });
        auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, dtype_traits<std::uint8_t>::ty)};
        auto [scale2, zero_point2] {compute_quant_config_from_data_naive(data_in.data(), numel, std::numeric_limits<std::uint8_t>::max()>>1)};
        ASSERT_NEAR(scale, scale2, 1e-6);
        ASSERT_EQ(zero_point, zero_point2);
    }
}
