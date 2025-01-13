#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace quant {
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
            std::int32_t zero_point
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
                std::span<const float> in {};
                std::span<std::uint8_t> out {};
            } payload {};
            std::thread thread {};
        };

        alignas(cache_line) volatile bool m_interrupt {};
        alignas(cache_line) std::uint64_t m_phase {};
        alignas(cache_line) std::uint64_t m_num_completed {};
        std::vector<worker> m_workers {};
        std::condition_variable m_cv {};
        std::mutex m_mtx {};

        auto worker_fn(worker& worker) -> void;
        auto kickoff_workers(
            std::span<const float> in,
            std::span<std::uint8_t> out,
            float scale,
            std::int32_t zero_point
        ) -> void;
    };
}
