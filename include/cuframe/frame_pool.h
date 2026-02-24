#pragma once

#include "cuframe/device_buffer.h"

#include <memory>
#include <vector>

namespace cuframe {

class FramePool;

// returns buffer to pool instead of freeing gpu memory
struct PoolDeleter {
    FramePool* pool = nullptr;
    void operator()(DeviceBuffer* buf) const;
};

using PooledBuffer = std::unique_ptr<DeviceBuffer, PoolDeleter>;

// pre-allocated pool of fixed-size gpu buffers. acquire() lends a buffer
// wrapped in a PooledBuffer that auto-returns on destruction.
// all PooledBuffers must be destroyed before the pool.
// not thread-safe. used internally by Decoder — relies on NVDEC calling
// display callbacks synchronously within cuvidParseVideoData.
class FramePool {
public:
    FramePool(size_t frame_size_bytes, int count);
    ~FramePool();

    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;

    // borrow a buffer. returns null PooledBuffer if exhausted.
    PooledBuffer acquire();

    int available() const;
    int capacity() const;

private:
    friend struct PoolDeleter;
    void release(DeviceBuffer* buf);

    size_t frame_size_ = 0;
    std::vector<std::unique_ptr<DeviceBuffer>> all_buffers_;
    std::vector<DeviceBuffer*> free_list_;
};

} // namespace cuframe
