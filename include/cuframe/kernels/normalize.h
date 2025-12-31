#pragma once
#include <cuda_runtime.h>

namespace cuframe {

struct NormParams {
    float scale[3];  // per-channel: 1.0 / (255.0 * std)
    float bias[3];   // per-channel: -mean / std
};

// pre-compute scale+bias from mean/std arrays.
// input expected in [0, 255] range.
// output: out[c] = in[c] * scale[c] + bias[c] = (in[c]/255 - mean[c]) / std[c]
NormParams make_norm_params(const float mean[3], const float std[3]);

// ImageNet defaults: mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
extern const NormParams IMAGENET_NORM;

// apply per-channel normalize to float32 RGB planar (3 × H × W).
// in-place if src_ptr == dst_ptr.
void normalize(
    const float* src_ptr,
    float* dst_ptr,
    int width, int height,
    const NormParams& params,
    cudaStream_t stream = nullptr
);

} // namespace cuframe
