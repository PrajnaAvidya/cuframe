#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"

namespace cuframe {

// all-in-one: NV12 → bilinear resize → color convert → normalize → float32 RGB planar.
// output is 3 × resize.dst_h × resize.dst_w floats in planar layout (R, G, B planes).
// no intermediate buffers allocated.
void fused_nv12_to_tensor(
    const uint8_t* nv12_ptr,
    float* rgb_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const ResizeParams& resize,
    const ColorMatrix& color,
    const NormParams& norm,
    bool bgr = false,
    bool is_10bit = false,
    cudaStream_t stream = nullptr
);

} // namespace cuframe
