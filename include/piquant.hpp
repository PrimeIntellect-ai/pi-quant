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
    enum class round_mode {
        nearest,
        stochastic,

        count_
    };

    enum class reduce_op {
        set, // output[i] = dequantize(input[i])
        add, // output[i] += qdeuantize(input[i])

        count_
    };

    // All supported data types for quantization and dequantization. Order matters.
    enum class dtype {
        f32 = 0,
        bf16,

        uint2, // 2-bit unsigned int
        uint4, // 4-bit unsigned int
        uint8, // 8-bit unsigned int (uint8_t)

        count_
    };
    static_assert(static_cast<std::underlying_type_t<dtype>>(dtype::count_) <= 0xff);
    static_assert(static_cast<std::underlying_type_t<dtype>>(dtype::f32) == 0);
    static_assert(static_cast<std::underlying_type_t<dtype>>(dtype::bf16) == 1);

    struct uint2_t final {
        using packed_storage = std::uint8_t;
        packed_storage bits;

        constexpr uint2_t() noexcept : bits{} {}
        constexpr uint2_t(int u8) noexcept : bits{static_cast<packed_storage>(u8)} {}
        constexpr auto operator == (uint2_t rhs) const noexcept -> bool { return bits == rhs.bits; }
        constexpr auto operator != (uint2_t rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr auto operator == (packed_storage rhs) const noexcept -> bool { return bits == rhs; }
        constexpr auto operator != (packed_storage rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr explicit operator std::uint8_t() const noexcept { return bits; }
        constexpr explicit operator std::int64_t() const noexcept { return bits; }
    };

    struct uint4_t final {
        using packed_storage = std::uint8_t;
        packed_storage bits;

        constexpr uint4_t() noexcept : bits {} {}
        constexpr uint4_t(int u8) noexcept : bits {static_cast<packed_storage>(u8)} {}
        constexpr auto operator == (uint4_t rhs) const noexcept -> bool { return bits == rhs.bits; }
        constexpr auto operator != (uint4_t rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr auto operator == (packed_storage rhs) const noexcept -> bool { return bits == rhs; }
        constexpr auto operator != (packed_storage rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr explicit operator std::uint8_t() const noexcept { return bits; }
        constexpr explicit operator std::int64_t() const noexcept { return bits; }
    };

    using fp32_t = float; // IEEE 754 binary 32

    // Google Brain Float 16
    struct bfp16_t final {
        using packed_storage = std::uint16_t;
        packed_storage bits;

        constexpr bfp16_t() noexcept : bits {} {}
        constexpr bfp16_t(fp32_t s) noexcept {
            auto u32 {std::bit_cast<std::uint32_t>(s)};
            if ((u32 & 0x7fffffff) > 0x7f800000) bits = u32>>16|64; // Force quiet NaN
            else bits = (u32 + (0x7fff + ((u32>>16)&1)))>>16;
        }
        constexpr auto operator == (bfp16_t rhs) const noexcept -> bool { return bits == rhs.bits; }
        constexpr auto operator != (bfp16_t rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr auto operator == (packed_storage rhs) const noexcept -> bool { return bits == rhs; }
        constexpr auto operator != (packed_storage rhs) const noexcept -> bool { return !(*this == rhs); }
        constexpr explicit operator fp32_t() const noexcept { return std::bit_cast<fp32_t>(static_cast<std::uint32_t>(bits)<<16); }

        constexpr auto operator + (bfp16_t rhs) const noexcept -> bfp16_t {
            return {static_cast<fp32_t>(*this) + static_cast<fp32_t>(rhs)};
        }
        constexpr auto operator += (bfp16_t rhs) noexcept -> bfp16_t& {
            *this = *this + rhs;
            return *this;
        }
        constexpr auto operator - (bfp16_t rhs) const noexcept -> bfp16_t {
            return {static_cast<fp32_t>(*this) - static_cast<fp32_t>(rhs)};
        }
        constexpr auto operator -= (bfp16_t rhs) noexcept -> bfp16_t& {
            *this = *this - rhs;
            return *this;
        }
        constexpr auto operator * (bfp16_t rhs) const noexcept -> bfp16_t {
            return {static_cast<fp32_t>(*this) * static_cast<fp32_t>(rhs)};
        }
        constexpr auto operator *= (bfp16_t rhs) noexcept -> bfp16_t& {
            *this = *this * rhs;
            return *this;
        }
        constexpr auto operator / (bfp16_t rhs) const noexcept -> bfp16_t {
            return {static_cast<fp32_t>(*this) / static_cast<fp32_t>(rhs)};
        }
        constexpr auto operator /= (bfp16_t rhs) noexcept -> bfp16_t& {
            *this = *this / rhs;
            return *this;
        }
    };

    static_assert(sizeof(uint2_t) == 1);
    static_assert(sizeof(uint4_t) == 1);
    static_assert(sizeof(bfp16_t) == 2);

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
        dtype_info{.name="f32", .stride=sizeof(fp32_t), .bit_size=8*sizeof(fp32_t), .flags=dtype_flags::is_float|dtype_flags::is_signed},                                                  // f32
        dtype_info{.name="bf16", .stride=sizeof(bfp16_t), .bit_size=8*sizeof(bfp16_t), .flags=dtype_flags::is_float|dtype_flags::is_signed},
        dtype_info{.name="uint2", .stride=sizeof(std::uint8_t), .bit_size=2,  .flags=dtype_flags::is_quant|dtype_flags::is_int|dtype_flags::is_packed},                            // uint2
        dtype_info{.name="uint4", .stride=sizeof(std::uint8_t), .bit_size=4,  .flags=dtype_flags::is_quant|dtype_flags::is_int|dtype_flags::is_packed},                            // uint4
        dtype_info{.name="uint8", .stride=sizeof(std::uint8_t), .bit_size=8,  .flags=dtype_flags::is_quant|dtype_flags::is_int},                                                   // uint8
    };
    static_assert([]() -> bool {
        for (auto&& info : dtype_infos) {
            if (!info.bit_size || info.bit_size & (info.bit_size-1)) return false; // bit_size must be a power of two
            if (!((info.flags & dtype_flags::is_float) ^ (info.flags & dtype_flags::is_int))) return false; // Either is_fp32_t or is_int must be set, but not both
        }
        return true;
    }());
    [[nodiscard]] constexpr auto dtype_info_of(dtype dtype) noexcept -> const dtype_info& { return dtype_infos[static_cast<std::size_t>(dtype)]; }

    template <typename> struct dtype_limits final {};

    template<> struct dtype_limits<fp32_t> final {
        static constexpr fp32_t min {-std::numeric_limits<fp32_t>::max()}; // Referes to the smallest, normal, finite number, so it's like std::numeric_limits<float>::lowest()
        static constexpr fp32_t max {std::numeric_limits<fp32_t>::max()};
    };
    template<> struct dtype_limits<bfp16_t> final {
        static constexpr bfp16_t min {0xFF7F}; // Referes to the smallest, normal, finite number, so it's like std::numeric_limits<float>::lowest()
        static constexpr bfp16_t max {0x7F7F};
    };
    template<> struct dtype_limits<uint2_t> final {
        static constexpr std::uint8_t min {0};
        static constexpr std::uint8_t max {3};
    };
    template<> struct dtype_limits<uint4_t> final {
        static constexpr std::uint8_t min {0};
        static constexpr std::uint8_t max {15};
    };
    template<> struct dtype_limits<std::uint8_t> final {
        static constexpr std::uint8_t min {0};
        static constexpr std::uint8_t max {255};
    };

    template <typename T> concept is_float_type = std::is_floating_point_v<T> || std::is_same_v<T, bfp16_t>;
    template <typename T> concept is_quant_type = std::is_integral_v<T> || std::is_same_v<uint2_t, T> || std::is_same_v<uint4_t, T>;;
    template <typename T> concept is_dtype = is_float_type<T> || is_quant_type<T>;
    template <typename T> requires is_dtype<T> struct dtype_traits final {};

    template <> struct dtype_traits<fp32_t> { static constexpr dtype type_code {dtype::f32}; };
    template <> struct dtype_traits<bfp16_t> { static constexpr dtype type_code {dtype::bf16}; };
    template <> struct dtype_traits<uint2_t> { static constexpr dtype type_code {dtype::uint2}; };
    template <> struct dtype_traits<uint4_t> { static constexpr dtype type_code {dtype::uint4}; };
    template <> struct dtype_traits<std::uint8_t> { static constexpr dtype type_code {dtype::uint8}; };

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
            fp32_t scale,
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
            fp32_t scale,
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
            fp32_t scale,
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
            fp32_t scale,
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
            fp32_t scale,
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
            fp32_t scale,
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

        [[nodiscard]] auto compute_quant_config_from_data(std::span<const fp32_t> x, dtype quant_dst_dtype) const -> std::pair<fp32_t, std::int64_t>;
        [[nodiscard]] auto compute_quant_config_from_data(std::span<const bfp16_t> x, dtype quant_dst_dtype) const -> std::pair<fp32_t, std::int64_t>;

        class pimpl;

        enum class command_type {
            quant, // out[i] = quantize(in[i])
            dequant, // out[i] = dequantize(in[i])
            quant_dequant // out[i] = dequantize(quantize(in[i])))
        };

        struct quant_descriptor final {
            command_type type {command_type::quant};
            const std::byte* in {};
            std::byte* out {};
            std::int64_t numel {};
            fp32_t scale{};
            std::int64_t zero_point {};
            dtype dt_in {};
            dtype dt_out {};
            round_mode rounding {};
            reduce_op reducing {};
        };

    private:
        std::shared_ptr<pimpl> m_pimpl;
    };
}
