// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#include "../piquant_internal.hpp"

using namespace piquant;

// Xorshift 128 plus PRNG (scalar) used for stochastic rounding.
// Generates a canonical float ∈ [0, 1) using a 64-bit state.
struct xs128p_state final {
    std::uint64_t p1 {};
    std::uint64_t p2 {};

    constexpr xs128p_state(std::uint64_t p1, std::uint64_t p2) noexcept : p1{p1}, p2{p2} {}

    [[nodiscard]] auto PIQUANT_AINLINE operator ()() noexcept -> std::uint64_t {
        std::uint64_t s1 {p1};
        std::uint64_t s0 {p2};
        p1 = s0;
        s1 ^= s1<<23;
        p2 = s1^s0^(s1>>18)^(s0>>5);
        return p2 + s0;
    }

    [[nodiscard]] auto PIQUANT_AINLINE canonical() noexcept -> float {
        static constexpr auto bias_scale {1.0f/static_cast<float>(0x800000)};
        std::uint64_t y {~0u & (*this)()};
        return (bias_scale*(static_cast<float>(y>>9) + 0.5f));
    }

    [[nodiscard]] auto PIQUANT_AINLINE bounded(std::array<std::uint32_t, 2> bounds) noexcept -> std::array<std::uint32_t, 2> {
        auto [b1, b2] {bounds}; // [0, bound1), [0, bound2)
        std::uint64_t y {(*this)()};
        return {
            static_cast<std::uint32_t>(((y&~0u)*b1)>>32),
            static_cast<std::uint32_t>(((y>>32)*b2)>>32)
        };
    }
};
