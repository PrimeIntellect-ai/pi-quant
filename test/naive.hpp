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
    const std::span<const piquant::f32> in,
    const std::span<piquant::quint8> out,
    const double scale,
    const std::int32_t zero_point
) noexcept -> void { /* Original implementation */
    const std::size_t numel {in.size()};
    const auto* const p_in {in.data()};
    auto* const p_out {out.data()};
    const double inv_scale {1.0 / scale};
    piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
    for (std::size_t i {}; i < numel; ++i) {
        std::int32_t quant_val = static_cast<std::int32_t>(std::round(p_in[i] * inv_scale)) + zero_point;
        p_out[i] = static_cast<piquant::quint8>(std::clamp(quant_val, 0, 255));
    }
}

inline auto q4_naive(
    const std::span<const piquant::f32> in,
    const std::span<piquant::quint4> out,
    const double scale,
    const std::int32_t zero_point
) noexcept -> void {
    const std::size_t numel {in.size()};
    const std::size_t out_numel {(numel + 1) / 2};
    const double inv_scale {1.0 / scale};
    piquant_assert(out_numel == out.size(), "input and output spans must have the same length, but %zu != %zu", out_numel, out.size());
    const auto f = [=](float x) noexcept -> std::uint8_t {
        return std::clamp<int>(std::round(x * inv_scale) + zero_point, 0, 0xf);
    };
    for (std::size_t i {}; i < out_numel; ++i) {
        std::uint8_t hi = f(in[2 * i])     & 15;
        std::uint8_t lo = f(in[2 * i + 1]) & 15;
        out[i] = static_cast<piquant::quint4>((hi << 4) | lo);
    }
}
