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

    struct kernel_registry final {
        auto (*quant_kernel)(
          const void* x,
          void* o,
          std::int64_t numel,
          const context::quant_descriptor& desc
        ) noexcept -> void;
        auto (*quant_config_kernel_f32)(std::span<const float> x) noexcept -> std::array<float, 2>;
        auto (*quant_config_kernel_f64)(std::span<const double> x) noexcept -> std::array<double, 2>;
    };
}
