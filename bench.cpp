#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include "quant.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.hpp"

auto q8_naive(
    const std::span<const float> in,
    const std::span<std::uint8_t> out,
    const double scale,
    const std::int32_t zero_point,
    const std::size_t nt
) noexcept -> void { /* Original implementation */
    assert(in.size() == out.size());
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    const float inv_scale {static_cast<float>(1.0 / scale)};
    const std::size_t n_threads {std::max(1u, std::thread::hardware_concurrency())};
    const std::size_t chunk_size = numel / n_threads;
    std::vector<std::jthread> threads {};
    auto quantize_chunk = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int32_t quant_val = static_cast<int32_t>(std::round(p_in[i] * inv_scale)) + zero_point;
            p_out[i] = static_cast<uint8_t>(std::clamp(quant_val, 0, 255));
        }
    };
    for (int i = 0; i < nt - 1; ++i) {
        int64_t start = i*chunk_size;
        int64_t end = (i + 1)*chunk_size;
        threads.emplace_back(quantize_chunk, start, end);
    }
    threads.emplace_back(quantize_chunk, (n_threads - 1)*chunk_size, numel);
}

auto main() -> int {
    const std::size_t nt {std::max(1u, std::thread::hardware_concurrency())};
    volatile std::size_t numel {1'000'000'000}; // 4 GiB -> 1 GiB
    std::vector<float> data_in {};
    std::vector<std::uint8_t> data_out {};
    data_in.resize(numel);
    data_out.resize(numel);
    std::random_device rd {};
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dist {-1.0f, 1.0f};
    std::ranges::generate(data_in, [&] { return dist(gen); });

    ankerl::nanobench::Bench bench {};

    bench.title("NAIVE")
        .unit("reduce")
        .minEpochIterations(10)
        .relative(true);
    bench.performanceCounters(true);
    bench.run("NAIVE", [&] {
        q8_naive(data_in, data_out, 1.0, 0, nt);
    });

    ankerl::nanobench::doNotOptimizeAway(data_in.data());
    ankerl::nanobench::doNotOptimizeAway(data_out.data());

    bench.title("OPT")
        .unit("reduce")
        .minEpochIterations(10)
        .relative(true);
    bench.performanceCounters(true);
    bench.run("OPT", [&] {
        quant::f32_q8(data_in, data_out, 1.0, 0, nt);
    });

    ankerl::nanobench::doNotOptimizeAway(data_in.data());
    ankerl::nanobench::doNotOptimizeAway(data_out.data());
    return 0;
}
