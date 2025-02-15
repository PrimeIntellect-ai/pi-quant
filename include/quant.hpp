/* Core C++ 20 API */

#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <thread>
#include <condition_variable>
#include <optional>

namespace quant {
    struct prng_state final { // Mersenne-Twister 64
        std::uint32_t remaining {};
        std::uint32_t next {};
        std::array<std::uint32_t, 624> state {};

        explicit constexpr prng_state(std::uint32_t seed) noexcept {
            state[0] = seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            next = 0;
            remaining = 1;
        }

        [[nodiscard]] constexpr auto gen_canonical() -> float { // returns ξ ∈ [0, 1)
            if (--remaining <= 0) {
                remaining = 624;
                next = 0;
                uint32_t y, i;
                for (i = 0; i < 624-397; ++i) {
                    y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                    state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                for (; i < 624-1; ++i) {
                    y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                    state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                y = (state[624-1] & 0x80000000u) | (state[0] & 0x7fffffffu);
                state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
            }
            uint32_t y = state[next++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >> 18;
            return (1.f/static_cast<float>(1 << 23)*(static_cast<float>(y>>9) + 0.5f));
        }
    };

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

        auto quantize_uint8(
            std::span<const float> in,
            std::span<std::uint8_t> out,
            float scale,
            std::int32_t zero_point,
            round_mode mode
        ) -> void;

        auto quantize_uint4(
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
            enum {
                q_i8,
                q_i4
            } format {q_i8};
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
            alignas(cache_line) payload pl;
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
            enum class amd64_cpu_caps {
                none=0,
                sse_4_2,
                avx2,
                avx512,

                num_
            } cpu_caps {};
        #endif

        auto kickoff_workers(const op_info& info) -> void;
        auto barrier() -> void;
    };
}
