#pragma once

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include "piquant.hpp"
#include "../src/piquant_internal.hpp"

[[nodiscard]] static constexpr auto prng_canonical(piquant::prng_state& p) -> float { // returns ξ ∈ [0, 1)
    auto& remaining {p.remaining};
    auto& next {p.next};
    auto& state {p.state};
    if (--remaining <= 0) {
        remaining = 624;
        next = 0;
        uint32_t y, i;
        for (i = 0; i < 624-397; ++i) {
            y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
            state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
        }
        for (; i < 624-1; ++i) {
            y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
            state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
        }
        y = (state[624-1] & 0x80000000u) | (state[0] & 0x7fffffffu);
        state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
    }
    uint32_t y = state[next++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    return (1.f/static_cast<float>(1<<23)*(static_cast<float>(y>>9) + 0.5f));
}

template <typename IN, typename OUT, const piquant::round_mode RND> requires requires {
    requires std::is_floating_point_v<IN>;
    requires std::is_integral_v<OUT> || piquant::is_int4<OUT>;
}
auto quantize_naive(
    const std::span<IN> x,
    const std::span<OUT> o,
    const double scale,
    const std::int32_t zero_point
) noexcept -> void { /* Original implementation */
    const double inv_scale {1.0 / scale};
    if constexpr (RND == piquant::round_mode::nearest) {
        const auto Q{[&](const IN x) noexcept -> OUT {
            const double rnd {std::round(static_cast<double>(x) * inv_scale)};
            const auto integral {static_cast<std::int64_t>(rnd) + zero_point};
            return static_cast<OUT>(std::clamp<decltype(integral)>(integral, piquant::dtype_limits<OUT>::min, piquant::dtype_limits<OUT>::max));
        }};
        if constexpr (piquant::is_int4<OUT>)
            for (std::size_t i {}; i < (x.size()+1)>>1; ++i)
                o[i] = static_cast<OUT>((static_cast<std::underlying_type_t<OUT>>(Q(x[(i<<1)]))&15)<<4|static_cast<std::underlying_type_t<OUT>>(Q(x[(i<<1)+1]))&15);
        else
            for (std::int64_t i {}; i < x.size(); ++i)
                o[i] = Q(x[i]);
    } else {
        piquant::prng_state prng {9'3'2002};
        const auto Q{[&](const IN x) noexcept -> OUT {
            double rnd {x * inv_scale};
            const double dec {std::abs(rnd - std::trunc(rnd))};
            const double xi {prng_canonical(prng)};
            double adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const auto integral {static_cast<std::int64_t>(rnd) + zero_point};
            return static_cast<OUT>(std::clamp<decltype(integral)>(integral, piquant::dtype_limits<OUT>::min, piquant::dtype_limits<OUT>::max));
        }};
        if constexpr (piquant::is_int4<OUT>)
            for (std::size_t i {}; i < (x.size()+1)>>1; ++i)
                o[i] = static_cast<OUT>((static_cast<std::underlying_type_t<OUT>>(Q(x[(i<<1)]))&15)<<4|static_cast<std::underlying_type_t<OUT>>(Q(x[(i<<1)+1]))&15);
        else
            for (std::int64_t i {}; i < x.size(); ++i)
                o[i] = Q(x[i]);
    }
}

inline auto quantize_naive_4bit(
    const std::span<const float> in,
    const std::span<piquant::uint4_t> out,
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
        out[i] = static_cast<piquant::uint4_t>((hi << 4) | lo);
    }
}

template <typename T> requires std::is_floating_point_v<T>
[[nodiscard]] auto compute_quant_config_from_data(const T* p, std::int64_t numel, std::int64_t tmax) -> std::pair<T, std::int64_t> {
    if (!numel) [[unlikely]] return {0.0, 0.0};
    auto mean {static_cast<T>(std::accumulate(p, p+numel, 0.0) / static_cast<T>(numel))};
    auto sq_delta {static_cast<T>(std::transform_reduce(
        p, p+numel,
        0.0,
        std::plus{},
        [mean](const T value) noexcept -> T {
            T delta {value - mean};
            return delta*delta;
        }
    ))};
    const auto std {static_cast<T>(std::sqrt(sq_delta / static_cast<T>(numel-1)))};
    const auto scale {static_cast<T>(12.0f*std/static_cast<T>(tmax))};
    const std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean/scale))};
    return {scale, zp};
}
