// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#ifndef QUANT_KERNEL_IMPL
#error "Kernel impl is not defined"
#endif
#include <iostream>
#include <piquant.hpp>
#include "../piquant_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace piquant {
    struct kernel_registry;
}

#define concat(a, b) a ## b
#define impl_namespace(a, b) piquant::concat(a, _impl)

namespace impl_namespace(QUANT_KERNEL_IMPL, _) {

    // Include order matters, implementations are cloned per specialized compilation unit
    #include "prng.inl"
    #include "kernels_specialized.inl"
    #include "quantize.inl"
    #include "dequantize.inl"

    template <typename In, typename Out, const round_mode RoundMode, const reduce_op ReduceOp>
    static auto PIQUANT_HOT requant_generic(
      const void* in,
      void* out,
      std::int64_t numel,
      fp32_t scale,
      std::int64_t zp
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<In*>(out)};
        fp32_t inv_scale {1.0f / scale};
        if constexpr (ReduceOp == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<Out, In>(scale, zp, quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp));
            return;
        }
        if constexpr (ReduceOp == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<Out, In>(scale, zp, quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp));
            return;
        }
    }
};

namespace piquant {
    using quant_fn = auto (*)(const void*, void*, std::int64_t, fp32_t, std::int64_t) noexcept -> void;

    template <typename Src, typename Dst, round_mode M>
    [[nodiscard]] consteval auto quant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<Src, Dst, M>; }

    template <typename Src, typename Dst, reduce_op R>
    [[nodiscard]] consteval auto dequant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::dequant_generic<Dst, Src, R>; }

    template <typename Src, typename Dst, round_mode M, reduce_op R>
    [[nodiscard]] consteval auto requant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::requant_generic<Src, Dst, M, R>; }

    template <typename...> struct type_set {};

    template <typename TL> struct type_set_size;
    template <typename... Ts> struct type_set_size<type_set<Ts...>>  : std::integral_constant<std::size_t, sizeof...(Ts)> {};

    using quant_types = type_set<uint2_t, uint4_t, uint8_t>;

    using fp_types = type_set<fp32_t, bfp16_t>;

    template <typename Src, round_mode M, typename TL> struct make_quant_row;
    template <typename Src, round_mode M, typename... Dst>
    struct make_quant_row<Src, M, type_set<Dst...>> {
        static constexpr std::array<quant_fn, type_set_size<fp_types>::value + sizeof...(Dst)> value =
            {nullptr, nullptr, quant_entry<Src, Dst, M>()...};
    };

    template <typename Src, reduce_op R, typename TL> struct make_dequant_row;
    template <typename Src, reduce_op R, typename... Dst>
    struct make_dequant_row<Src, R, type_set<Dst...>> {
        static constexpr std::array<quant_fn, type_set_size<fp_types>::value + sizeof...(Dst)> value =
            {nullptr, nullptr, dequant_entry<Src, Dst, R>()...};
    };

    template <typename Src, round_mode M, reduce_op R, typename TL> struct make_requant_row;
    template <typename Src, round_mode M, reduce_op R, typename... Dst>
    struct make_requant_row<Src, M, R, type_set<Dst...>> {
        static constexpr std::array<quant_fn, type_set_size<fp_types>::value + sizeof...(Dst)> value =
            {nullptr, nullptr, requant_entry<Src, Dst, M, R>()...};
    };

    template <round_mode M, reduce_op R, typename FPSrcSet> struct make_requant_block;

    template <round_mode M, reduce_op R, typename... Src>
    struct make_requant_block<M, R, type_set<Src...>> {
        static constexpr std::array<std::array<quant_fn, type_set_size<fp_types>::value + type_set_size<quant_types>::value>, sizeof...(Src)> value {
            make_requant_row<Src, M, R, quant_types>::value...
        };
    };

    // 3D Dispatch table for quantization kernels. Order matters.
    static constexpr std::array quant_functions {
        std::array {
            std::array {
                make_quant_row<fp32_t, round_mode::nearest, quant_types>::value,
                make_quant_row<bfp16_t, round_mode::nearest, quant_types>::value,
            },
        },
        std::array {
            std::array {
                make_quant_row<fp32_t, round_mode::stochastic, quant_types>::value,
                make_quant_row<bfp16_t, round_mode::stochastic, quant_types>::value
            },
        }
    };

    // 3D Dispatch table for dequantization kernels. Order matters.
    static constexpr std::array dequant_functions {
        std::array {
            std::array {
                make_dequant_row<fp32_t, reduce_op::set, quant_types>::value,
                make_dequant_row<bfp16_t, reduce_op::set, quant_types>::value
            },
        },
        std::array {
            std::array {
                make_dequant_row<fp32_t, reduce_op::add, quant_types>::value,
                make_dequant_row<bfp16_t, reduce_op::add, quant_types>::value
            },
        }
    };

    // 4D Dispatch table for requantization kernels. Order matters.
    static constexpr std::array requant_functions {
        std::array{
            make_requant_block<round_mode::nearest, reduce_op::set, fp_types>::value,
            make_requant_block<round_mode::nearest, reduce_op::add, fp_types>::value
        },
        std::array{
            make_requant_block<round_mode::stochastic, reduce_op::set, fp_types>::value,
            make_requant_block<round_mode::stochastic, reduce_op::add, fp_types>::value
        }
    };

    static void dispatch_quantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
        piquant_assert2(dt_out.flags & dtype_flags::is_quant);
        const auto& stubs_round_mode {quant_functions[static_cast<std::size_t>(desc.rounding)]};
        const auto& stubs_dtype_fp {stubs_round_mode[static_cast<std::size_t>(desc.dt_in)]};
        auto* kernel {stubs_dtype_fp[static_cast<std::size_t>(desc.dt_out)]};
        piquant_assert(kernel != nullptr, "invalid quantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static void dispatch_dequantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(dt_in.flags & dtype_flags::is_quant);
        piquant_assert2(!(dt_out.flags & dtype_flags::is_quant));
        const auto& stubs_reduce_mode {dequant_functions[static_cast<std::size_t>(desc.reducing)]};
        const auto& stubs_dtype_fp {stubs_reduce_mode[static_cast<std::size_t>(desc.dt_out)]};
        auto* kernel {stubs_dtype_fp[static_cast<std::size_t>(desc.dt_in)]};
        piquant_assert(kernel != nullptr, "invalid dequantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static void dispatch_requantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        using enum dtype;
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
        piquant_assert2(dt_out.flags & dtype_flags::is_quant);
        const auto& stubs_round_mode {requant_functions[static_cast<std::size_t>(desc.rounding)]};
        const auto& stubs_reduce_op {stubs_round_mode[static_cast<std::size_t>(desc.reducing)]};
        const auto& stubs_fp {stubs_reduce_op[static_cast<std::size_t>(desc.dt_in)]};
        auto* kernel {stubs_fp[static_cast<std::size_t>(desc.dt_out)]};
        piquant_assert(kernel != nullptr, "invalid requantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static auto PIQUANT_HOT quantize_dispatch(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) noexcept -> void {
        switch (desc.type) {
            case context::command_type::quant: dispatch_quantize(in, out, range, desc); return;
            case context::command_type::dequant: dispatch_dequantize(in, out, range, desc); return;
            case context::command_type::quant_dequant: dispatch_requantize(in, out, range, desc); return;
            default: panic("invalid quantization command type: %d", static_cast<int>(desc.type));
        }
    }

    auto QUANT_KERNEL_IMPL() noexcept -> kernel_registry {
        return kernel_registry {
            .quant_kernel = &quantize_dispatch,
            .find_min_max_float32 = &impl_namespace(QUANT_KERNEL_IMPL, _)::find_min_max_f32,
            .find_min_max_bfloat16 = &impl_namespace(QUANT_KERNEL_IMPL, _)::find_min_max_bf16,
        };
    }
}
