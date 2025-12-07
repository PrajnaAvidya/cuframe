#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace cuframe {

// raii wrapper for device memory. uses cudaMallocAsync/cudaFreeAsync
// when a stream is provided, falls back to cudaMalloc/cudaFree otherwise.
// caller owns the stream — must outlive the buffer if stream-ordered.
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size_bytes, cudaStream_t stream = nullptr);
    ~DeviceBuffer();

    // move only, no copy
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() const;
    size_t size() const;

private:
    void free();

    void* ptr_ = nullptr;
    size_t size_ = 0;
    cudaStream_t stream_ = nullptr;
};

} // namespace cuframe
