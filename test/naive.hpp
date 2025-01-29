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
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    if (out.size() != in.size()) {
        std::cerr << "uint8 output span must have the same length as input span, but has " << out.size() << ", required: " << in.size() << std::endl;
        std::abort();
    }
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
) noexcept -> void {
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    const std::size_t out_numel {(numel + 1) / 2};
    if (out.size() != out_numel) {
        std::cerr << "int4 output span must have (input.size() + 1) / 2 length, but has " << out.size() << ", required: " << out_numel << std::endl;
        std::abort();
    }
    const float inv_scale {static_cast<float>(1.0 / scale)};
    const std::size_t n_threads {std::max(1u, std::thread::hardware_concurrency())};
    const std::size_t chunk_size = (numel + n_threads - 1) / n_threads; // Ensures proper division
    std::vector<std::thread> threads {};
    auto quantize_chunk = [&](int64_t start, int64_t end) {
        if (start >= end) return;
        std::int64_t vmel = end - start;
        std::size_t packed_length = vmel / 2;
        std::size_t out_start = start / 2;
        auto* output = p_out + out_start;
        auto* input = p_in + start;
        for (std::size_t i = 0; i < packed_length; ++i) {
            auto q1 = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(input[i*2] * inv_scale)) + zero_point, 0, 0xf));
            auto q2 = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(input[i*2 + 1] * inv_scale)) + zero_point, 0, 0xf));
            output[i] = (q1 << 4) | q2;
        }
        if (end == numel && (numel % 2 != 0)) {
            p_out[out_numel - 1] = static_cast<std::uint8_t>(
                (std::clamp(static_cast<std::int32_t>(std::round(input[vmel - 1] * inv_scale)) + zero_point, 0, 0xf) & 0xf) << 4
            );
        }
    };
    for (int i = 0; i < nt - 1; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = std::min((i + 1) * chunk_size, numel);
        threads.emplace_back(quantize_chunk, start, end);
    }
    threads.emplace_back(quantize_chunk, (n_threads - 1) * chunk_size, numel);

    for (auto& t : threads) t.join();
}
