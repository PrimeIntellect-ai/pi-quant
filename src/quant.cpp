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
    context::context(std::size_t num_threads) {
        num_threads = std::max<std::size_t>(1, num_threads);
        m_workers.resize(num_threads);
        for (std::int64_t ti {}; auto& worker : m_workers) {
            worker.payload.ti = ti++;
            worker.payload.tc = static_cast<std::int64_t>(num_threads);
            worker.phase = 0;
            worker.thread = std::thread(&context::worker_fn, this, std::ref(worker));
        }
    }

    auto context::worker_fn(worker& worker) -> void {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock {m_mtx};
                m_cv.wait(lock, [&]() noexcept -> bool { return m_interrupt || m_phase > worker.phase; });
                if (m_interrupt) [[unlikely]] break;
                worker.phase = m_phase;
            }
            const auto* bx {worker.payload.in.data()};
            auto* br {worker.payload.out.data()};
            std::int64_t tc {worker.payload.tc};
            std::int64_t ti {worker.payload.ti};
            std::int64_t numel {static_cast<std::int64_t>(worker.payload.in.size())};
            std::int64_t chunk {(numel + tc - 1)/tc};
            std::int64_t ra {chunk*ti};
            std::int64_t rb {std::min(ra + chunk, numel)};
            if (rb <= ra) continue;
            std::int64_t vmel {rb - ra};
            const auto* px {bx + ra};
            auto* pr {br + ra};
            float scale {worker.payload.scale};
            std::int32_t zp {worker.payload.zero_point};
            f32_q8_generic(px, pr, vmel, scale, zp);
            {
                std::unique_lock<std::mutex> lock {m_mtx};
                if (++m_num_completed == m_workers.size())
                    m_cv.notify_all();
            }
        }
    }

    auto context::kickoff_workers(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point
    ) -> void {
        {
            std::unique_lock<std::mutex> lock {m_mtx};
            for (auto& worker : m_workers) {
                worker.payload.in = in;
                worker.payload.out = out;
                worker.payload.scale = scale;
                worker.payload.zero_point = zero_point;
            }
            ++m_phase;
            m_num_completed = 0;
        }
        m_cv.notify_all();
        {
            std::unique_lock<std::mutex> lock {m_mtx};
            m_cv.wait(lock, [&] noexcept -> bool { return m_num_completed == m_workers.size(); });
        }
    }

    context::~context() {
        std::unique_lock<std::mutex> lock {m_mtx};
        m_interrupt = true;
        ++m_phase;
        lock.unlock();
        m_cv.notify_all();
        for (auto& worker : m_workers) worker.thread.join();
        m_workers.clear();
    }

    auto context::quantize_int8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point
    ) -> void {
        kickoff_workers(in, out, scale, zero_point);
    }
}
