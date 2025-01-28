#pragma once

#include <array>
#include <cstdint>

namespace quant {
    struct prng_state final { // Mersenne-Twister 64
        std::uint32_t remaining {};
        std::uint32_t next {};
        std::array<std::uint32_t, 624> state {};

        explicit constexpr prng_state(std::uint32_t seed) noexcept {
            state[0] = seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            next = 0;
            remaining = 1;
        }

        [[nodiscard]] constexpr auto gen_canonical() -> float { // returns ξ ∈ [0, 1)
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
            return (1.f/static_cast<float>(1 << 23)*(static_cast<float>(y>>9) + 0.5f));
        }
    };
}
