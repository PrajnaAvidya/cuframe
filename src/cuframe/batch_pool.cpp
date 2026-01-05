#include "cuframe/batch_pool.h"

namespace cuframe {

BatchPool::BatchPool(int pool_size, int max_batch, int c, int h, int w) {
    slots_.reserve(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->batch = std::make_unique<GpuFrameBatch>(max_batch, c, h, w);
        slots_.push_back(std::move(slot));
    }
}

BatchPool::~BatchPool() = default;

std::shared_ptr<GpuFrameBatch> BatchPool::try_acquire() {
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        bool expected = true;
        if (slots_[i]->free.compare_exchange_strong(expected, false)) {
            auto* raw = slots_[i]->batch.get();
            return std::shared_ptr<GpuFrameBatch>(raw,
                [this, i](GpuFrameBatch*) { release(i); });
        }
    }
    return nullptr;
}

std::shared_ptr<GpuFrameBatch> BatchPool::acquire() {
    // fast path: try without locking
    auto result = try_acquire();
    if (result) return result;

    // slow path: wait for a slot to free up
    std::unique_lock lock(mutex_);
    for (;;) {
        cv_.wait(lock, [this] {
            for (auto& slot : slots_)
                if (slot->free.load(std::memory_order_acquire))
                    return true;
            return false;
        });
        // retry after wakeup — another thread may have grabbed it
        result = try_acquire();
        if (result) return result;
    }
}

void BatchPool::release(int slot_idx) {
    slots_[slot_idx]->free.store(true, std::memory_order_release);
    cv_.notify_one();
}

int BatchPool::capacity() const {
    return static_cast<int>(slots_.size());
}

int BatchPool::available() const {
    int count = 0;
    for (auto& slot : slots_)
        if (slot->free.load(std::memory_order_acquire))
            ++count;
    return count;
}

} // namespace cuframe
