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
        struct worker final {
            struct {
                alignas(std::hardware_destructive_interference_size) std::int64_t ti {}; // thread index
                alignas(std::hardware_destructive_interference_size) std::int64_t tc {}; // thread count
                alignas(std::hardware_destructive_interference_size) std::uint64_t phase {};
                float scale {};
                std::int32_t zero_point {};
                std::span<const float> in {};
                std::span<std::uint8_t> out {};
            } payload {};
            std::thread thread {};
        };

        alignas(std::hardware_destructive_interference_size) volatile bool m_interrupt {};
        alignas(std::hardware_destructive_interference_size) std::uint64_t m_phase {};
        alignas(std::hardware_destructive_interference_size) std::uint64_t m_num_completed {};
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
