#pragma once

#include <array>
#include <span>
#include <cstdint>

#ifdef _MSC_VER
#ifdef QUANT_BUILD_SHARED
#define QUANT_EXPORT __declspec(dllexport)
#else
#define QUANT_EXPORT
#endif
#else
#define QUANT_EXPORT __attribute__((visibility("default")))
#endif

#include <pithreadpool/threadpool.hpp>

namespace piquant {
    // computes and returns {scale, zero_point} derived from the data's mean and stddev.
    [[nodiscard]] QUANT_EXPORT std::pair<float, std::int64_t> compute_quant_config_from_data(std::span<const float> x, std::int64_t tmax);
    [[nodiscard]] QUANT_EXPORT std::pair<double, std::int64_t> compute_quant_config_from_data(std::span<const double> x, std::int64_t tmax);

    using quant_task_future = pi::threadpool::TaskFuture<pi::threadpool::void_t>;

    enum class round_mode {
        nearest,
        stochastic
    };

    enum class reduce_op {
        set, // output[i] = dequantize(input[i])
        add, // output[i] += qdeuantize(input[i])
    };

    enum class dtype {
        uint4,
        int4,
        uint8,
        int8,
        uint16,
        int16,
        uint32,
        int32,
        uint64,
        int64,
        f32,
        f64,

        num_
    };
    static_assert(static_cast<std::underlying_type_t<dtype>>(dtype::num_) <= 0xff);

    enum class uint4_t : std::uint8_t {};
    enum class int4_t : std::uint8_t {};

    struct dtype_info final {
        std::size_t stride;
        std::size_t bit_size;
        bool is_quant;
    };

    constexpr std::array dtype_infos {
        dtype_info{.stride=1, .bit_size=4, .is_quant=true}, // uint4
        dtype_info{.stride=1, .bit_size=4, .is_quant=true}, // int4
        dtype_info{.stride=1, .bit_size=8, .is_quant=true}, // uint8
        dtype_info{.stride=1, .bit_size=8, .is_quant=true}, // int8
        dtype_info{.stride=2, .bit_size=16, .is_quant=true}, // uint16
        dtype_info{.stride=2, .bit_size=16, .is_quant=true}, // int16
        dtype_info{.stride=4, .bit_size=32, .is_quant=true}, // uint32
        dtype_info{.stride=4, .bit_size=32, .is_quant=true}, // int32
        dtype_info{.stride=8, .bit_size=64, .is_quant=true}, // uint64
        dtype_info{.stride=8, .bit_size=64, .is_quant=true}, // int64
        dtype_info{.stride=4, .bit_size=32, .is_quant=false}, // f32
        dtype_info{.stride=8, .bit_size=64, .is_quant=false}, // f64
    };
    [[nodiscard]] constexpr auto dtype_info_of(dtype dtype) noexcept -> const dtype_info & { return dtype_infos[static_cast<std::size_t>(dtype)]; }

    template <typename T> struct dtype_limits final {
        static constexpr T min{std::numeric_limits<T>::min()};
        static constexpr T max{std::numeric_limits<T>::max()};
    };
    template<> struct dtype_limits<uint4_t> final {
        static constexpr std::uint8_t min{0};
        static constexpr std::uint8_t max{0xf};
    };
    template<> struct dtype_limits<int4_t> final {
        static constexpr std::int8_t min{-0x8};
        static constexpr std::int8_t max{0x7};
    };

    template<typename T> concept is_int4 = std::is_same_v<T, uint4_t> || std::is_same_v<T, int4_t>;
    template<typename T> concept is_dtype = std::is_arithmetic_v<T> || is_int4<T>;
    template<typename T> requires is_dtype<T> struct dtype_traits final {};

    template<> struct dtype_traits<uint4_t> { static constexpr dtype ty = dtype::uint4; };
    template<> struct dtype_traits<int4_t> { static constexpr dtype ty = dtype::int4; };
    template<> struct dtype_traits<std::int8_t> { static constexpr dtype ty = dtype::int8; };
    template<> struct dtype_traits<std::uint8_t> { static constexpr dtype ty = dtype::uint8; };
    template<> struct dtype_traits<std::int16_t> { static constexpr dtype ty = dtype::int16; };
    template<> struct dtype_traits<std::uint16_t> { static constexpr dtype ty = dtype::uint16; };
    template<> struct dtype_traits<std::int32_t> { static constexpr dtype ty = dtype::int32; };
    template<> struct dtype_traits<std::uint32_t> { static constexpr dtype ty = dtype::uint32; };
    template<> struct dtype_traits<std::int64_t> { static constexpr dtype ty = dtype::int64; };
    template<> struct dtype_traits<std::uint64_t> { static constexpr dtype ty = dtype::uint64; };
    template<> struct dtype_traits<float> { static constexpr dtype ty = dtype::f32; };
    template<> struct dtype_traits<double> { static constexpr dtype ty = dtype::f64; };

    class QUANT_EXPORT context final {
    public:
        explicit context(std::size_t num_threads, std::size_t task_queue_size = 8192);
        context(const context&) = delete;
        context(context&&) = delete;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&&) -> context& = delete;
        ~context();

        [[nodiscard]] auto quantize(
            std::span<const std::byte> in,
            dtype dtype_in,
            std::span<std::byte> out,
            dtype dtype_out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) const -> quant_task_future;

        template<typename IN, typename OUT> requires requires {
            requires is_dtype<IN>;
            requires is_dtype<OUT>;
            dtype_info_of(dtype_traits<IN>::ty).is_quant == true;
            dtype_info_of(dtype_traits<OUT>::ty).is_quant == false;
        }
        [[nodiscard]] auto quantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) -> quant_task_future {
            return quantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<IN>::ty,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<OUT>::ty,
                scale,
                zero_point,
                mode
            );
        }

        [[nodiscard]] auto dequantize(
            std::span<const std::byte> in,
            dtype dtype_in,
            std::span<std::byte> out,
            dtype dtype_out,
            float scale,
            std::int32_t zero_point,
            reduce_op op
        ) const -> quant_task_future;

        template<typename IN, typename OUT> requires requires {
            requires is_dtype<IN>;
            requires is_dtype<OUT>;
            dtype_info_of(dtype_traits<IN>::ty).is_quant == false;
            dtype_info_of(dtype_traits<OUT>::ty).is_quant == true;
        }
        [[nodiscard]] auto dequantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int32_t zero_point,
            reduce_op op
        ) -> quant_task_future {
            return dequantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<IN>::ty,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<OUT>::ty,
                scale,
                zero_point,
                op
            );
        }

        [[nodiscard]] auto quantize_dequantize_fused(
            std::span<const std::byte> in,
            dtype dtype_in_out,
            std::span<std::byte> out,
            dtype quant_type,
            float scale,
            std::int32_t zero_point,
            round_mode mode,
            reduce_op op
        ) const -> quant_task_future;

        template<typename INOUT, typename QUANT> requires requires {
            requires is_dtype<INOUT>;
            dtype_info_of(dtype_traits<INOUT>::ty).is_quant == false;
            dtype_info_of(dtype_traits<QUANT>::ty).is_quant == true;
        }
        [[nodiscard]] auto quantize_dequantize_fused_generic(
            std::span<const INOUT> in,
            std::span<INOUT> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode,
            reduce_op op
        ) {
            return quantize_dequantize_fused(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<INOUT>::ty,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<QUANT>::ty,
                scale,
                zero_point,
                mode,
                op
            );
        }

        class pimpl;

        enum class command_type {
            quant, // out[i] = quantize(in[i])
            dequant, // out[i] = dequantize(in[i])
            quant_dequant // out[i] = dequantize(quantize(in[i])))
        };

        struct quant_descriptor final {
            command_type type{command_type::quant};
            const std::byte* in{};
            std::byte* out{};
            std::int64_t numel{};
            float scale{};
            std::int32_t zero_point{};
            dtype dt_in{};
            dtype dt_out{};
            round_mode rnd_mode{};
            reduce_op reduce{};
        };

    private:
        std::shared_ptr<pimpl> m_pimpl;
    };
}
