/* Core C++ 20 API */

#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <thread>
#include <condition_variable>
#include <optional>
#include <variant>

#ifdef _MSC_VER
#define QUANT_EXPORT __declspec(dllexport)
#else
#define QUANT_EXPORT __attribute__((visibility("default")))
#endif

namespace piquant {
    // computes and returns {scale, zero_point} derived from the data's mean and stddev.
    [[nodiscard]] extern auto compute_quant_config_from_data(std::span<const float> x) -> std::pair<float, std::int32_t>;

    /* Aborts with a formatted message. Because not all tested C++ compilers support std::format, C-style formatting is used for now. Should be replaced later. Pulling in fmt::format just for abort seems a bit too much... */
    [[noreturn]] extern auto panic(const char* msg, ...) -> void;

    #define QUANT_STRINGIZE2(x) #x
    #define QUANT_STRINGIZE(x) QUANT_STRINGIZE2(x)
    #define QUANT_SRC_NAME __FILE__ ":" QUANT_STRINGIZE(__LINE__)

    #define piquant_assert(expr, msg, ...) \
        if ((!(expr))) [[unlikely]] { \
            ::piquant::panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
        }
    #define piquant_assert2(expr) piquant_assert(expr, "")

    enum class round_mode {
        nearest,
        stochastic
    };

    #ifdef __x86_64__
        enum class amd64_cpu_caps {
            none=0,
            sse_4_2,
            avx2,
            avx512,

            num_
        };
    #endif

    enum class reduce_op {
        set, // output[i] = dequantize(input[i])
        add, // output[i] += qdeuantize(input[i])
    };

    struct prng_state final {
        // Mersenne-Twister 64
        std::uint32_t remaining {};
        std::uint32_t next {};
        std::array<std::uint32_t, 624> state {};

        constexpr prng_state(const std::uint32_t seed) {
            state[0] = seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            next = 0;
            remaining = 1;
        }
    };

    enum class dtype {
        f32,
        uint8,
        uint4,

        num_
    };

    using f32 = float;
    enum class quint8 : std::uint8_t {};
    enum class quint4 : std::uint8_t {};

    struct dtype_info final {
        std::size_t sto_size;
        std::size_t bit_size;
    };

    constexpr std::array dtype_infos {
        dtype_info{sizeof(f32), sizeof(f32)<<3},
        dtype_info{sizeof(quint8), sizeof(quint8)<<3},
        dtype_info{sizeof(quint4), 4}
    };
    [[nodiscard]] constexpr auto dtype_info_of(dtype dtype) noexcept -> const dtype_info& {
        return dtype_infos[static_cast<std::size_t>(dtype)];
    }

    template <typename  T>
    concept is_dtype = std::is_same_v<T, f32>
        || std::is_same_v<T, quint8>
        || std::is_same_v<T, quint4>;

    template <typename T> requires is_dtype<T>
    struct dtype_traits final {};
    template<> struct dtype_traits<f32> final { static constexpr auto ty{dtype::f32}; };
    template<> struct dtype_traits<quint8> final { static constexpr auto ty{dtype::uint8}; };;
    template<> struct dtype_traits<quint4> final { static constexpr auto ty{dtype::uint4}; };

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
            std::int32_t zero_point,
            round_mode mode
        ) const -> void;

        template <typename IN, typename OUT> requires (is_dtype<IN> && is_dtype<OUT>)
        auto quantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) {
            quantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size()},
                dtype_traits<IN>::ty,
                {reinterpret_cast<std::byte*>(out.data()), out.size()},
                dtype_traits<OUT>::ty,
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
            std::int32_t zero_point,
            reduce_op op
        ) const -> void;

        template <typename IN, typename OUT> requires (is_dtype<IN> && is_dtype<OUT>)
        auto dequantize_generic(
            std::span<const IN> in,
            std::span<OUT> out,
            float scale,
            std::int32_t zero_point,
            reduce_op op
        ) {
            dequantize(
                {reinterpret_cast<const std::byte*>(in.data()), in.size()},
                dtype_traits<IN>::ty,
                {reinterpret_cast<std::byte*>(out.data()), out.size()},
                dtype_traits<OUT>::ty,
                scale,
                zero_point,
                op
            );
        }

        class pimpl;

    private:
        std::shared_ptr<pimpl> m_pimpl;
    };
}
