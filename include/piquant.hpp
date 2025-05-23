#pragma once

#include <array>
#include <span>
#include <cstdint>
#include <string_view>
#include <memory>

#ifdef _MSC_VER
#ifdef QUANT_BUILD_SHARED
#define QUANT_EXPORT __declspec(dllexport)
#else
#define QUANT_EXPORT
#endif
#else
#define QUANT_EXPORT __attribute__((visibility("default")))
#endif

namespace piquant {
    // (All types except u/int4) Amount of standard deviations/sigmas (left and right of 0) to use for the quantization range
    static constexpr double stddev_scale {12.0};

    // (u/int4 only) Amount of standard deviations/sigmas (left and right of 0) to use for the quantization range
    static constexpr double stddev_scale_int4 {2.7};

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

    struct uint4_t final {
        std::uint8_t u8;
        constexpr uint4_t() noexcept : u8 {} {}
        constexpr uint4_t(int u8) noexcept : u8 {static_cast<std::uint8_t>(u8)} {}
        constexpr auto operator == (std::uint8_t y) const noexcept -> bool { return this->u8 == y; }
        constexpr auto operator != (std::uint8_t y) const noexcept -> bool { return !(*this == y); }
        constexpr auto operator == (uint4_t y) const noexcept -> bool { return this->u8 == y.u8; }
        constexpr auto operator != (uint4_t y) const noexcept -> bool { return !(*this == y); }
        constexpr auto pack(std::uint8_t lo, std::uint8_t hi) noexcept -> void {
            u8 = static_cast<std::uint8_t>((lo & 15) | ((hi & 15) << 4));
        }
        [[nodiscard]] constexpr auto unpack() const noexcept -> std::array<std::uint8_t, 2> {
            return {static_cast<std::uint8_t>(u8 & 15), static_cast<std::uint8_t>(u8 >> 4)};
        }
    };

    struct int4_t final {
        std::int8_t u8;
        constexpr int4_t() noexcept : u8 {} {}
        constexpr int4_t(int u8) noexcept : u8 {static_cast<std::int8_t>(u8)} {}
        constexpr auto operator == (std::int8_t u8) const noexcept -> bool { return this->u8 == u8; }
        constexpr auto operator != (std::int8_t y) const noexcept -> bool { return !(*this == y); }
        constexpr auto operator == (int4_t y) const noexcept -> bool { return this->u8 == y.u8; }
        constexpr auto operator != (int4_t y) const noexcept -> bool { return !(*this == y); }
        constexpr auto pack(std::int8_t lo, std::int8_t hi) noexcept -> void {
            u8 = static_cast<std::int8_t>((lo & 15) | ((hi & 15) << 4));
        }
        [[nodiscard]] constexpr auto unpack() const noexcept -> std::array<std::int8_t, 2> {
            constexpr auto snex4 {[](std::int8_t x) noexcept -> std::int8_t { return x & 0x8 ? static_cast<std::int8_t>(x|0xF0) : x; }};
            return {(snex4(u8 & 15)), (snex4(u8 >> 4))};
        }
    };

    static_assert(sizeof(uint4_t) == 1);
    static_assert(sizeof(int4_t) == 1);

    struct dtype_flags final {
        enum $ {
            none = 0,
            is_quant = 1<<0,
            is_float = 1<<1,
            is_int = 1<<2,
            is_signed = 1<<3,
            is_packed = 1<<4,
        };
    };

    struct dtype_info final {
        std::string_view name;
        std::size_t stride;
        std::size_t bit_size;
        std::underlying_type_t<dtype_flags::$> flags;
    };

    constexpr std::array dtype_infos {
        dtype_info{.name="uint4", .stride=1, .bit_size=4,  .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_packed},                            // uint4
        dtype_info{.name="int4", .stride=1, .bit_size=4,  .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_packed+dtype_flags::is_signed},      // int4
        dtype_info{.name="uint8", .stride=1, .bit_size=8,  .flags=dtype_flags::is_quant+dtype_flags::is_int},                                                   // uint8
        dtype_info{.name="int8", .stride=1, .bit_size=8,  .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_signed},                             // int8
        dtype_info{.name="uint16", .stride=2, .bit_size=16, .flags=dtype_flags::is_quant+dtype_flags::is_int},                                                  // uint16
        dtype_info{.name="int16", .stride=2, .bit_size=16, .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_signed},                            // int16
        dtype_info{.name="uint32", .stride=4, .bit_size=32, .flags=dtype_flags::is_quant+dtype_flags::is_int},                                                  // uint32
        dtype_info{.name="int32", .stride=4, .bit_size=32, .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_signed},                            // int32
        dtype_info{.name="uint64", .stride=8, .bit_size=64, .flags=dtype_flags::is_quant+dtype_flags::is_int},                                                  // uint64
        dtype_info{.name="int64", .stride=8, .bit_size=64, .flags=dtype_flags::is_quant+dtype_flags::is_int+dtype_flags::is_signed},                            // int64
        dtype_info{.name="f32", .stride=4, .bit_size=32, .flags=dtype_flags::is_float+dtype_flags::is_signed},                                                  // f32
        dtype_info{.name="f64", .stride=8, .bit_size=64, .flags=dtype_flags::is_float+dtype_flags::is_signed},                                                  // f64
    };
    static_assert([]() -> bool {
        for (auto&& info : dtype_infos) {
            if (!info.bit_size) return false;
            if (!((info.flags & dtype_flags::is_float) ^ (info.flags & dtype_flags::is_int))) return false;
        }
        return true;
    }());
    [[nodiscard]] constexpr auto dtype_info_of(dtype dtype) noexcept -> const dtype_info& { return dtype_infos[static_cast<std::size_t>(dtype)]; }

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

    template<> struct dtype_traits<uint4_t> { static constexpr dtype type_code = dtype::uint4; };
    template<> struct dtype_traits<int4_t> { static constexpr dtype type_code = dtype::int4; };
    template<> struct dtype_traits<std::int8_t> { static constexpr dtype type_code = dtype::int8; };
    template<> struct dtype_traits<std::uint8_t> { static constexpr dtype type_code = dtype::uint8; };
    template<> struct dtype_traits<std::int16_t> { static constexpr dtype type_code = dtype::int16; };
    template<> struct dtype_traits<std::uint16_t> { static constexpr dtype type_code = dtype::uint16; };
    template<> struct dtype_traits<std::int32_t> { static constexpr dtype type_code = dtype::int32; };
    template<> struct dtype_traits<std::uint32_t> { static constexpr dtype type_code = dtype::uint32; };
    template<> struct dtype_traits<std::int64_t> { static constexpr dtype type_code = dtype::int64; };
    template<> struct dtype_traits<std::uint64_t> { static constexpr dtype type_code = dtype::uint64; };
    template<> struct dtype_traits<float> { static constexpr dtype type_code = dtype::f32; };
    template<> struct dtype_traits<double> { static constexpr dtype type_code = dtype::f64; };

    class QUANT_EXPORT context final {
    public:
        explicit context(std::size_t num_threads);
        context(const context&) = delete;
        context(context&&) = delete;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&&) -> context& = delete;
        ~context();

        auto quantize(
            std::span<const std::byte> in,
            dtype dtype_in,
            std::span<std::byte> out,
            dtype dtype_out,
            float scale,
            std::int64_t zero_point,
            round_mode mode
        ) const -> void;

        template<typename IN, typename OUT> requires requires {
            requires is_dtype<IN>;
            requires is_dtype<OUT>;
            dtype_info_of(dtype_traits<IN>::type_code).flags & dtype_flags::is_quant;
            !(dtype_info_of(dtype_traits<OUT>::type_code).flags & dtype_flags::is_quant);
        }
        auto quantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int64_t zero_point,
            round_mode mode
        ) -> void {
            quantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<IN>::type_code,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<OUT>::type_code,
                scale,
                zero_point,
                mode
            );
        }

        auto dequantize(
            std::span<const std::byte> in,
            dtype dtype_in,
            std::span<std::byte> out,
            dtype dtype_out,
            float scale,
            std::int64_t zero_point,
            reduce_op op
        ) const -> void;

        template<typename IN, typename OUT> requires requires {
            requires is_dtype<IN>;
            requires is_dtype<OUT>;
            !(dtype_info_of(dtype_traits<IN>::type_code).flags & dtype_flags::is_quant);
            dtype_info_of(dtype_traits<OUT>::type_code).flags & dtype_flags::is_quant;
        }
        auto dequantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int64_t zero_point,
            reduce_op op
        ) -> void {
            dequantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<IN>::type_code,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<OUT>::type_code,
                scale,
                zero_point,
                op
            );
        }

        auto quantize_dequantize_fused(
            std::span<const std::byte> in,
            dtype dtype_in_out,
            std::span<std::byte> out,
            dtype quant_type,
            float scale,
            std::int64_t zero_point,
            round_mode mode,
            reduce_op op
        ) const -> void;

        template<typename INOUT, typename QUANT> requires requires {
            requires is_dtype<INOUT>;
            !(dtype_info_of(dtype_traits<INOUT>::type_code).flags & dtype_flags::is_quant);
            dtype_info_of(dtype_traits<QUANT>::type_code).flags & dtype_flags::is_quant;
        }
        auto quantize_dequantize_fused_generic(
            std::span<const INOUT> in,
            std::span<INOUT> out,
            float scale,
            std::int64_t zero_point,
            round_mode mode,
            reduce_op op
        ) -> void {
            quantize_dequantize_fused(
                {reinterpret_cast<const std::byte*>(in.data()), in.size_bytes()},
                dtype_traits<INOUT>::type_code,
                {reinterpret_cast<std::byte*>(out.data()), out.size_bytes()},
                dtype_traits<QUANT>::type_code,
                scale,
                zero_point,
                mode,
                op
            );
        }

        [[nodiscard]] auto compute_quant_config_from_data(std::span<const float> x, dtype quant_dst_dtype) const -> std::pair<float, std::int64_t>;
        [[nodiscard]] auto compute_quant_config_from_data(std::span<const double> x, dtype quant_dst_dtype) const -> std::pair<float, std::int64_t>;

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
            std::int64_t zero_point{};
            dtype dt_in{};
            dtype dt_out{};
            round_mode rnd_mode{};
            reduce_op reduce{};
        };

    private:
        std::shared_ptr<pimpl> m_pimpl;
    };
}
