#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <thread>
#include <condition_variable>
#include <optional>

#include "prng.hpp"

namespace quant {
    enum class round_mode : bool {
        nearest = true,
        stochastic = false
    };

    class context final {
    public:
        explicit context(std::size_t num_threads);
        context(const context&) = delete;
        context(context&&) = delete;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&&) -> context& = delete;
        ~context();

        auto quantize_int8(
            std::span<const float> in,
            std::span<std::uint8_t> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) -> void;

    private:
        static constexpr std::size_t cache_line {
            #ifdef __cpp_lib_hardware_interference_size
                std::hardware_destructive_interference_size
            #else
                        64
            #endif
        };

        struct op_info final {
            const float* in {};
            std::uint8_t* out {};
            std::int64_t numel {};
            float scale {};
            std::int32_t zero_point {};
            round_mode rnd_mode {};
        };

        struct payload {
            prng_state prng;
            std::int64_t ti {}; // thread index
            std::int64_t tc {}; // thread count
            std::uint64_t phase {};

            explicit constexpr payload(const std::uint32_t seed) noexcept : prng{seed} {}
        };

        struct worker final {
            context* ctx;
            alignas(cache_line) payload payload;
            alignas(cache_line) op_info op {};
            std::optional<std::thread> thread {};

            explicit worker(context& ctx, std::int64_t ti, std::int64_t tc);
            worker(const worker&) = delete;
            worker(worker&&) = default;
            auto operator = (const worker&) -> worker& = delete;
            auto operator = (worker&&) -> worker& = default;
            ~worker() = default;

            [[nodiscard]] auto await_work() -> bool;
            auto entry() -> void;
            auto exec_and_broadcast() -> void;
        };

        alignas(cache_line) volatile bool m_interrupt {};
        alignas(cache_line) std::uint64_t m_phase {};
        alignas(cache_line) std::atomic_int64_t m_num_completed {};
        std::vector<worker> m_workers {};
        std::condition_variable m_cv {};
        std::mutex m_mtx {};
        std::atomic_size_t m_workers_online {};

        #ifdef __x86_64__
            bool m_sse42_supported : 1 {};
            bool m_avx2_supported : 1 {};
            bool m_avx512f_supported : 1 {};
        #endif

        auto kickoff_workers(
            std::span<const float> in,
            std::span<std::uint8_t> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) -> void;
        auto barrier() -> void;
    };
}
