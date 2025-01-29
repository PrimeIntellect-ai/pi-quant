#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include "../src/quant.hpp"
#include "naive.hpp"

#ifdef __x86_64__
[[nodiscard]] extern auto check_sse42_support() noexcept -> bool;
[[nodiscard]] extern auto check_avx2_support() noexcept -> bool;
[[nodiscard]] extern auto check_avx512f_support() noexcept -> bool;
#endif

auto main() -> int {
    const std::size_t nt {std::max(1u, std::thread::hardware_concurrency())};
    std::cout << "Num threads: " << nt << std::endl;
    #ifdef __x86_64__
        std::cout << "SSE 4.2? " << (check_sse42_support() ? "YES" : "NO") << std::endl;
        std::cout << "AVX-2? " << (check_avx2_support() ? "YES" : "NO") << std::endl;
        std::cout << "AVX-512 F? " << (check_avx512f_support() ? "YES" : "NO") << std::endl;
    #endif

    volatile std::size_t numel {1'000'000};
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

    q8_naive(data_in, data_out_naive, 1.0, 0, nt);
    quant::context ctx {nt};
    ctx.quantize_uint8(data_in, data_out, 1.0, 0, quant::round_mode::nearest);

    for (std::size_t i {}; i < numel; ++i) {
        auto a = data_out_naive[i];
        auto b = data_out[i];
        if (a != b) {
            std::cerr << "Mismatch at index " << i << ": " << static_cast<int>(a) << " != " << static_cast<int>(b) << std::endl;
            return -1;
        }
    }

    std::cout << "Test passed!" << std::endl;

    return 0;
}

