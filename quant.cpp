#include "quant.hpp"

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

#define impl generic
#include "q8_kernel.inl"
#undef impl

namespace quant {
    static auto has_avx2() noexcept -> bool { // Should return true for AMD "Excavator", Intel "Haswell" or later processors
        std::int32_t info[4] = {-1};
        #if (defined(__clang__) || defined(__GNUC__)) && defined(__cpuid)
            __cpuid(0, info[0], info[1], info[2], info[3]);
        #else
            __cpuid(info, 0);
        #endif
        if (info[0] < 7) return false;
        #if (defined(__clang__) || defined(__GNUC__)) && defined(__cpuid)
            __cpuid(1, info[0], info[1], info[2], info[3]);
        #else
            __cpuid(info, 1);
        #endif
        if ((info[2]&0x38081001) != 0x38081001) // We check for F16C, FMA3, AVX, OSXSAVE, SSSE4.1, and SSE3
            return false;
        #if defined(__clang__) || defined(__GNUC__)
            __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
        #else
            __cpuidex(info, 7, 0);
        #endif
        return (info[1]&0x20) == 0x20;
    }

    auto f32_q8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const double scale,
        const std::int32_t zero_point
    ) -> void {
        assert(in.size() == out.size());
        const std::size_t numel {in.size()};
        const auto* const p_in {in.data()};
        auto* const p_out {out.data()};
        const float inv_scale {static_cast<float>(1.0 / scale)};
        const std::size_t n_threads {std::max(1u, std::thread::hardware_concurrency())};
        const std::size_t chunk_size = numel / n_threads;
        std::vector<std::jthread> threads {};
        const auto quantize_chunk = [=](std::size_t start, std::size_t end) noexcept -> void {
            assert(end >= start);
            f32_q8_kernel_impl(p_in + start, p_out + start, std::abs(end - start), inv_scale, zero_point);
        };
        for (std::size_t i {}; i < n_threads-1; ++i) {
            std::size_t start = i*chunk_size;
            std::size_t end = (i + 1)*chunk_size;
            threads.emplace_back(quantize_chunk, start, end);
        }
        threads.emplace_back(quantize_chunk, (n_threads - 1)*chunk_size, numel);
        for (auto& thread : threads) {
            thread.join();
        }
    }
}
