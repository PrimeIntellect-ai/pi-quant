#pragma once

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

inline auto q8_naive(
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
    std::vector<std::thread> threads {};
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
    for (auto& t : threads)t.join();
}

inline auto q4_naive(
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
    std::memset(p_out, 0, numel>>1);
    const float inv_scale {static_cast<float>(1.0 / scale)};
    const std::size_t n_threads {std::max(1u, std::thread::hardware_concurrency())};
    const std::size_t chunk_size = numel / n_threads;
    std::vector<std::thread> threads {};
    auto quantize_chunk = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int32_t quant_val = static_cast<int32_t>(std::round(p_in[i] * inv_scale)) + zero_point;
            std::uint8_t nibble = static_cast<uint8_t>(std::clamp(quant_val, 0, 15));
            p_out[i/2] |= nibble << ((i % 2) * 4);
        }
    };
    for (int i = 0; i < nt - 1; ++i) {
        int64_t start = i*chunk_size;
        int64_t end = (i + 1)*chunk_size;
        threads.emplace_back(quantize_chunk, start, end);
    }
    threads.emplace_back(quantize_chunk, (n_threads - 1)*chunk_size, numel);
    for (auto& t : threads)t.join();
}
