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

    #ifdef _MSC_VER
    #define PIQUANT_HOT
    #define PIQUANT_AINLINE __forceinline
    #define PIQUANT_RESTRICT __restrict
    #else
    #define PIQUANT_HOT __attribute__((hot))
    #define PIQUANT_AINLINE __attribute__((always_inline)) inline
    #define PIQUANT_RESTRICT __restrict__
    #endif

    struct kernel_registry final {
        auto (*quant_kernel)(
          const void* x,
          void* o,
          std::int64_t numel,
          const context::quant_descriptor& desc
        ) noexcept -> void;
        auto (*find_min_max_float32)(std::span<const fp32_t> x) noexcept -> std::array<fp32_t, 2>;
        auto (*find_min_max_bfloat16)(std::span<const bfp16_t> x) noexcept -> std::array<fp32_t, 2>;
    };

    [[nodiscard]] constexpr auto packed_numel(std::size_t ne, const dtype_info& dto) noexcept -> std::size_t {
        std::size_t per_byte {8u / dto.bit_size};
        return (ne + per_byte-1)/per_byte;
    }
}
