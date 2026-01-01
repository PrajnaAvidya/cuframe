#pragma once
#include "cuframe/device_buffer.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace cuframe {

// contiguous NCHW float32 tensor in GPU memory.
// move-only, owns its buffer.
class GpuFrameBatch {
public:
    // allocate batch buffer for n frames of c × h × w float32
    GpuFrameBatch(int n, int c, int h, int w);

    GpuFrameBatch(GpuFrameBatch&&) noexcept = default;
    GpuFrameBatch& operator=(GpuFrameBatch&&) noexcept = default;
    GpuFrameBatch(const GpuFrameBatch&) = delete;
    GpuFrameBatch& operator=(const GpuFrameBatch&) = delete;

    float* data() const;
    float* frame(int i) const;

    int batch_size() const;
    int channels() const;
    int height() const;
    int width() const;
    size_t frame_size_bytes() const;  // C * H * W * sizeof(float)
    size_t total_size_bytes() const;  // N * C * H * W * sizeof(float)

private:
    DeviceBuffer buffer_;
    int n_, c_, h_, w_;
};

// copy N float32 planar frames into a pre-allocated batch.
// frame_ptrs[i] is a device pointer to frame i (c × h × w floats).
// host array of device pointers. async on given stream.
void batch_frames(
    GpuFrameBatch& batch,
    const float* const* frame_ptrs,
    int n,
    cudaStream_t stream = nullptr
);

} // namespace cuframe
