#pragma once

#include <cassert>
#include <numeric>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <span>

#include "piquant.hpp"
#include "../src/piquant_internal.hpp"

// Xorshift 128 plus PRNG (scalar) used for stochastic rounding.
// Generates a canonical float ∈ [0, 1) using a 64-bit state.
struct xs128p_state final {
   std::uint64_t p1 {};
   std::uint64_t p2 {};

   constexpr xs128p_state(std::uint64_t p1, std::uint64_t p2) noexcept : p1{p1}, p2{p2} {}

   [[nodiscard]] auto operator ()() noexcept -> std::uint64_t {
       std::uint64_t s1 {p1};
       std::uint64_t s0 {p2};
       p1 = s0;
       s1 ^= s1<<23;
       p2 = s1^s0^(s1>>18)^(s0>>5);
       return p2 + s0;
   }

   [[nodiscard]] auto canonical() noexcept -> float {
       static constexpr auto bias_scale {1.0f/static_cast<float>(0x800000)};
       std::uint64_t y {~0u & (*this)()};
       return (bias_scale*(static_cast<float>(y>>9) + 0.5f));
   }
};

static constinit xs128p_state s_sprng {0x123456789abcdef0, 0x0fedcba987654321};

[[nodiscard]] inline auto xs32_canonical() noexcept -> float {
    return s_sprng.canonical();
}

template <typename OUT> requires piquant::is_int4<OUT>
[[nodiscard]] static constexpr auto pack_nibbles(OUT x, OUT y) -> OUT {
    auto xi {x.u8};
    auto yi {y.u8};
    auto pa {static_cast<std::uint8_t>((0xf&xi)<<4|0xf&yi)};
    return OUT{pa};
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
        if constexpr (piquant::is_int4<OUT>) {
            std::int64_t numel_out = (x.size()+1) >> 1;
            for (std::int64_t i = 0, j = 0; j < numel_out; ++j, i += 2) {
                IN a = x[i];
                IN b = i+1 < x.size() ? x[i+1] : x[i];
                o[j] = pack_nibbles(Q(a), Q(b));
            }
        } else {
            for (std::int64_t i {}; i < x.size(); ++i) {
                o[i] = Q(x[i]);
            }
        }
    } else {
        const auto Q{[&](const IN x) noexcept -> OUT {
            double rnd {x * inv_scale};
            const double dec {std::abs(rnd - std::trunc(rnd))};
            const double xi {xs32_canonical()};
            double adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const auto integral {static_cast<std::int64_t>(rnd) + zero_point};
            return static_cast<OUT>(std::clamp<decltype(integral)>(integral, piquant::dtype_limits<OUT>::min, piquant::dtype_limits<OUT>::max));
        }};
        if constexpr (piquant::is_int4<OUT>)
            for (std::size_t i {}; i < (x.size()+1)>>1; ++i)
                o[i] = static_cast<OUT>(((Q(x[(i<<1)]).u8)&15)<<4|(Q(x[(i<<1)+1]).u8)&15);
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
[[nodiscard]] auto compute_quant_config_from_data_naive(const T* p, std::int64_t numel, std::int64_t tmax) -> std::pair<T, std::int64_t> {
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
    const auto scale {static_cast<T>(piquant::stddev_scale*std/static_cast<T>(tmax))};
    const std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean/scale))};
    return {scale, zp};
}
