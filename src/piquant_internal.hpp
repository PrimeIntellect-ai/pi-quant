#pragma once

#include "piquant.hpp"

namespace piquant {

    /* Aborts with a formatted message. Because not all tested C++ compilers support std::format, C-style formatting is used for now. Should be replaced later. Pulling in fmt::format just for abort seems a bit too much... */
    [[noreturn]] void panic(const char *msg, ...);

    #define QUANT_STRINGIZE2(x) #x
    #define QUANT_STRINGIZE(x) QUANT_STRINGIZE2(x)
    #define QUANT_SRC_NAME __FILE__ ":" QUANT_STRINGIZE(__LINE__)

    #define piquant_assert(expr, msg, ...) \
    if ((!(expr))) [[unlikely]] { \
    ::piquant::panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
    #define piquant_assert2(expr) piquant_assert(expr, "")

    #ifdef __x86_64__
        enum class amd64_cpu_caps {
            none=0,
            sse_4_2,
            avx2,
            avx512,

            num_
        };
    #endif

    struct prng_state final {
        // Mersenne-Twister 64
        std::uint32_t remaining{};
        std::uint32_t next{};
        std::array<std::uint32_t, 624> state{};

        constexpr prng_state(const std::uint32_t seed) {
            state[0] = seed;
            for (size_t i = 1; i < 624; ++i)
                state[i] = ((state[i - 1] ^ (state[i - 1] >> 30)) * 1812433253 + i) & ~0u;
            next = 0;
            remaining = 1;
        }
    };
}
