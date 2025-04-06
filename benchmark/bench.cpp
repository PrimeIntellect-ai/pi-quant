#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include <piquant.hpp>
#include "../test/naive.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.hpp"

auto main() -> int {
    const std::size_t nt {std::max(1u, std::thread::hardware_concurrency())};
    volatile std::size_t numel {1024*1024*1024/4};
    std::vector<float> data_in {};
    std::vector<std::uint8_t> data_out {};
    data_in.resize(numel);
    data_out.resize(numel);
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution dist {-1.0f, 1.0f};
    std::ranges::generate(data_in, [&] { return dist(gen); });

    ankerl::nanobench::Bench bench {};

    piquant::context ctx {nt};
    bench.run("quantize", [&] {
        auto [scale, zero_point] {ctx.compute_quant_config_from_data(data_in, piquant::dtype::uint8)};
        ankerl::nanobench::doNotOptimizeAway(zero_point);
        ankerl::nanobench::doNotOptimizeAway(scale);
    });
    return 0;
}
