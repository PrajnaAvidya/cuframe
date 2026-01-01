#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

GpuFrameBatch::GpuFrameBatch(int n, int c, int h, int w)
    : buffer_(static_cast<size_t>(n) * c * h * w * sizeof(float)),
      n_(n), c_(c), h_(h), w_(w) {}

float* GpuFrameBatch::data() const {
    return static_cast<float*>(buffer_.data());
}

float* GpuFrameBatch::frame(int i) const {
    return data() + static_cast<size_t>(i) * c_ * h_ * w_;
}

int GpuFrameBatch::batch_size() const { return n_; }
int GpuFrameBatch::channels() const { return c_; }
int GpuFrameBatch::height() const { return h_; }
int GpuFrameBatch::width() const { return w_; }

size_t GpuFrameBatch::frame_size_bytes() const {
    return static_cast<size_t>(c_) * h_ * w_ * sizeof(float);
}

size_t GpuFrameBatch::total_size_bytes() const {
    return static_cast<size_t>(n_) * frame_size_bytes();
}

void batch_frames(
    GpuFrameBatch& batch,
    const float* const* frame_ptrs,
    int n,
    cudaStream_t stream
) {
    for (int i = 0; i < n; ++i) {
        CUFRAME_CUDA_CHECK(cudaMemcpyAsync(
            batch.frame(i),
            frame_ptrs[i],
            batch.frame_size_bytes(),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
}

} // namespace cuframe
