#pragma once

#include <span>
#include <cstdint>

namespace quant {
    extern auto f32_q8(
        std::span<const float> in,
        std::span<std::uint8_t> out,
        double scale,
        std::int32_t zero_point,
        std::size_t nt
    ) -> void;
}
