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
    const std::int32_t zero_point
) noexcept -> void { /* Original implementation */
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    const double inv_scale {1.0 / scale};
    if (out.size() != in.size()) {
        std::cerr << "uint8 output span must have the same length as input span, but has " << out.size() << ", required: " << in.size() << std::endl;
        std::abort();
    }
    for (std::size_t i {}; i < numel; ++i) {
        int32_t quant_val = static_cast<int32_t>(std::round(p_in[i] * inv_scale)) + zero_point;
        p_out[i] = static_cast<uint8_t>(std::clamp(quant_val, 0, 255));
    }
}

inline auto q4_naive(
    const std::span<const float> in,
    const std::span<std::uint8_t> out,
    const double scale,
    const std::int32_t zero_point
) noexcept -> void {
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    const std::size_t out_numel {(numel + 1) / 2};
    const double inv_scale {1.0 / scale};
    if (out.size() != out_numel) {
        std::cerr << "int4 output span must have (input.size() + 1) / 2 length, but has " << out.size() << ", required: " << out_numel << std::endl;
        std::abort();
    }
    const auto f = [=](float x) noexcept -> std::uint8_t {
        return std::clamp<int>(std::round(x * inv_scale) + zero_point, 0, 0xf);
    };
    for (std::size_t i {}; i < out_numel; ++i) {
        std::uint8_t hi = f(in[2 * i])     & 15;
        std::uint8_t lo = f(in[2 * i + 1]) & 15;
        out[i] = (hi << 4) | lo;
    }
}
