#pragma once
#include "cuframe/gpu_frame_batch.h"
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace cuframe {

class BatchPool {
public:
    // pre-allocate pool_size batches, each max_batch x c x h x w float32
    BatchPool(int pool_size, int max_batch, int c, int h, int w);
    ~BatchPool();

    BatchPool(const BatchPool&) = delete;
    BatchPool& operator=(const BatchPool&) = delete;

    // blocks until a slot is free. returned shared_ptr auto-returns on drop.
    std::shared_ptr<GpuFrameBatch> acquire();

    // non-blocking. returns nullptr if exhausted.
    std::shared_ptr<GpuFrameBatch> try_acquire();

    int capacity() const;
    int available() const;

private:
    struct Slot {
        std::unique_ptr<GpuFrameBatch> batch;
        std::atomic<bool> free{true};
    };

    // heap-allocated slots — atomic<bool> is non-movable so can't store Slot in vector directly
    std::vector<std::unique_ptr<Slot>> slots_;
    std::mutex mutex_;
    std::condition_variable cv_;

    void release(int slot_idx);
};

} // namespace cuframe
