#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <thread>
#include <condition_variable>
#include <optional>
#include <mutex>

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

        struct worker final {
            struct {
                alignas(cache_line) std::int64_t ti {}; // thread index
                alignas(cache_line) std::int64_t tc {}; // thread count
                alignas(cache_line) std::uint64_t phase {};
                float scale {};
                std::int32_t zero_point {};
                round_mode mode {};
                const float* in {};
                std::uint8_t* out {};
                std::int64_t numel {};
                struct {
                    std::uint32_t remaining {};
                    std::uint32_t next {};
                    std::array<std::uint32_t, 624> state {};
                } prng {};
            } payload {};
            std::optional<std::thread> thread {};

            [[nodiscard]] auto await_work(context& ctx) -> bool;
            auto entry(context& ctx) -> void;
            auto exec_and_broadcast(context& ctx) -> void;
            auto prng_init(std::uint32_t seed) noexcept -> void;
            [[nodiscard]] auto prng_uniform(float min, float max) noexcept -> float;
        };

        alignas(cache_line) volatile bool m_interrupt {};
        alignas(cache_line) std::uint64_t m_phase {};
        alignas(cache_line) std::atomic_int m_num_completed {};
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
