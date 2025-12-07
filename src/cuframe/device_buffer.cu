#include "cuframe/device_buffer.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

DeviceBuffer::DeviceBuffer(size_t size_bytes, cudaStream_t stream)
    : size_(size_bytes), stream_(stream) {
    if (size_bytes == 0) return;

    if (stream_) {
        CUFRAME_CUDA_CHECK(cudaMallocAsync(&ptr_, size_bytes, stream_));
    } else {
        CUFRAME_CUDA_CHECK(cudaMalloc(&ptr_, size_bytes));
    }
}

DeviceBuffer::~DeviceBuffer() {
    free();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), stream_(other.stream_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.stream_ = nullptr;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;

    free();

    ptr_ = other.ptr_;
    size_ = other.size_;
    stream_ = other.stream_;

    other.ptr_ = nullptr;
    other.size_ = 0;
    other.stream_ = nullptr;

    return *this;
}

void* DeviceBuffer::data() const { return ptr_; }
size_t DeviceBuffer::size() const { return size_; }

void DeviceBuffer::free() {
    if (!ptr_) return;

    // destructor must not throw — ignore errors
    if (stream_) {
        cudaFreeAsync(ptr_, stream_);
    } else {
        cudaFree(ptr_);
    }

    ptr_ = nullptr;
    size_ = 0;
}

} // namespace cuframe
