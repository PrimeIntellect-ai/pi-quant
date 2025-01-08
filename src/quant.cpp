#include "quant.hpp"

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

#define Q8_KERNEL_IMPL f32_q8_generic
#include "q8_kernel.inl"
#undef Q8_KERNEL_IMPL

namespace quant {
    auto f32_q8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const double scale,
        const std::int32_t zero_point,
        std::size_t nt
    ) -> void {
        assert(in.size() == out.size());
        const std::size_t numel {in.size()};
        const auto* const p_in {in.data()};
        auto* const p_out {out.data()};
        const float inv_scale {static_cast<float>(1.0 / scale)};
        const std::size_t rpt = numel/nt; // Ceildiv to work-balance the remaining elements
        std::vector<std::thread> threads {};
        const auto Q = [=](std::size_t start, std::size_t end) noexcept -> void {
            assert(end >= start);
            f32_q8_generic(p_in + start, p_out + start, end-start, inv_scale, zero_point);
        };
        for (std::size_t i {}; i < nt; ++i) {
            std::size_t start = i*rpt;
            std::size_t end = std::min(numel, start+rpt);
            threads.emplace_back(Q, start, end);
        }
        for (auto& t : threads) t.join();
    }
}
