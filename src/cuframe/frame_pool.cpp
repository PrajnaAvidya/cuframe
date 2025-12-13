#include "cuframe/frame_pool.h"

namespace cuframe {

void PoolDeleter::operator()(DeviceBuffer* buf) const {
    if (buf && pool) pool->release(buf);
}

FramePool::FramePool(size_t frame_size_bytes, int count)
    : frame_size_(frame_size_bytes) {
    all_buffers_.reserve(count);
    free_list_.reserve(count);

    for (int i = 0; i < count; ++i) {
        all_buffers_.push_back(std::make_unique<DeviceBuffer>(frame_size_bytes));
        free_list_.push_back(all_buffers_.back().get());
    }
}

FramePool::~FramePool() = default;

PooledBuffer FramePool::acquire() {
    if (free_list_.empty()) return PooledBuffer(nullptr, PoolDeleter{this});

    DeviceBuffer* buf = free_list_.back();
    free_list_.pop_back();
    return PooledBuffer(buf, PoolDeleter{this});
}

void FramePool::release(DeviceBuffer* buf) {
    free_list_.push_back(buf);
}

int FramePool::available() const {
    return static_cast<int>(free_list_.size());
}

int FramePool::capacity() const {
    return static_cast<int>(all_buffers_.size());
}

} // namespace cuframe
