#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include "../src/quant.hpp"
#include "../test/naive.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.hpp"

auto main() -> int {
    const std::size_t nt {std::max(1u, std::thread::hardware_concurrency())>>1};
    volatile std::size_t numel {500'000'000}; // 4 GiB -> 1 GiB
    std::vector<float> data_in {};
    std::vector<std::uint8_t> data_out {};
    data_in.resize(numel);
    data_out.resize(numel);
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
    std::ranges::generate(data_in, [&] { return dist(gen); });

    ankerl::nanobench::Bench bench {};

    bench.title("int8 Quantization Benchmark")
        .unit("quant")
        .minEpochIterations(10)
        .relative(true);
    bench.performanceCounters(true);
    bench.run("NAIVE", [&] {
        q8_naive(data_in, data_out, 1.0, 0, nt);
    });

    ankerl::nanobench::doNotOptimizeAway(data_in.data());
    ankerl::nanobench::doNotOptimizeAway(data_out.data());

    quant::context ctx {nt};

    bench.run("OPTIMIZED", [&] {
        ctx.quantize_int8(data_in, data_out, 1.0, 0);
    });

    ankerl::nanobench::doNotOptimizeAway(data_in.data());
    ankerl::nanobench::doNotOptimizeAway(data_out.data());
    return 0;
}
