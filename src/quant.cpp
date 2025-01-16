#include "quant.hpp"

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

#ifdef __x86_64__

#endif

extern auto __attribute__((hot)) f32_q8_generic(
    const float* __restrict__ x,
    std::uint8_t* __restrict__ o,
    std::size_t n,
    float inv_scale,
    std::int32_t zero_point
) noexcept -> void;

namespace quant {
    context::context(std::size_t num_threads) {
        num_threads = std::max<std::size_t>(1, num_threads);
        m_workers.resize(num_threads);
        for (std::int64_t ti {0}; ti < m_workers.size(); ++ti) { // Initialize workers (main thread is worker 0)
            auto& worker {m_workers[ti]};
            worker.payload.ti = ti;
            worker.payload.tc = static_cast<std::int64_t>(num_threads);
            worker.payload.phase = 0;
            if (ti != 0)
                worker.thread = std::thread{&worker::entry, &worker, std::ref(*this)};
        }
        while (m_workers_online.load(std::memory_order_seq_cst) != num_threads-1)
            std::this_thread::yield();
    }

    auto context::worker::await_work(context& ctx) -> bool {
        std::unique_lock lock {ctx.m_mtx};
        ctx.m_cv.wait(lock, [&]() noexcept -> bool { return ctx.m_interrupt || ctx.m_phase > payload.phase; });
        if (ctx.m_interrupt) [[unlikely]] return false;
        payload.phase = ctx.m_phase;
        return true;
    }

    auto context::worker::entry(context& ctx) -> void {
        ctx.m_workers_online.fetch_add(1, std::memory_order_seq_cst);
        while (await_work(ctx)) [[likely]]
            exec_and_broadcast(ctx);
    }

    auto context::worker::exec_and_broadcast(context& ctx) -> void {
        const auto* const bx {payload.in.data()};
        auto* const br {payload.out.data()};
        const std::int64_t tc {payload.tc};
        const std::int64_t ti {payload.ti};
        const std::int64_t numel {static_cast<std::int64_t>(payload.in.size())};
        const std::int64_t chunk {(numel + tc - 1)/tc};
        const std::int64_t ra {chunk*ti};
        const std::int64_t rb {std::min(ra + chunk, numel)};
        if (rb > ra) [[likely]] {
            const std::int64_t vmel {rb - ra};
            const auto* const px {bx + ra};
            auto* const pr {br + ra};
            const float scale {payload.scale};
            const std::int32_t zp {payload.zero_point};
            f32_q8_generic(px, pr, vmel, scale, zp);
        }
        std::unique_lock lock {ctx.m_mtx};
        if (++ctx.m_num_completed == ctx.m_workers.size())
            ctx.m_cv.notify_all();
    }

    auto context::kickoff_workers(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        std::unique_lock lock {m_mtx};
        for (auto& worker : m_workers) {
            worker.payload.in = in;
            worker.payload.out = out;
            worker.payload.scale = scale;
            worker.payload.zero_point = zero_point;
            worker.payload.mode = mode;
        }
        ++m_phase;
        m_num_completed = 0;
    }

    auto context::barrier() -> void {
        std::unique_lock lock {m_mtx};
        m_cv.wait(lock, [&]() noexcept -> bool { return m_num_completed == m_workers.size(); });
    }

    context::~context() {
        std::unique_lock lock {m_mtx};
        m_interrupt = true;
        ++m_phase;
        lock.unlock();
        m_cv.notify_all();
        for (auto& worker : m_workers)
            if (worker.thread && worker.thread->joinable())
                worker.thread->join();
        m_workers.clear();
    }

    auto context::quantize_int8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        kickoff_workers(in, out, scale, zero_point, mode);
        m_cv.notify_all();
        worker& w0 {m_workers[0]}; // Main thread does work too
        w0.exec_and_broadcast(*this);
        barrier();
    }
}
