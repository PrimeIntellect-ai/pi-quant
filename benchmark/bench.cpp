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
    ctx.quantize_generic<float, std::uint8_t>(data_in, data_out, 0.2f, 127, piquant::round_mode::nearest);
    bench.run("dequantize", [&] {
        ctx.dequantize_generic<std::uint8_t, float>(data_out, data_in, 0.2f, 127, piquant::reduce_op::add);
    });
    return 0;
}
