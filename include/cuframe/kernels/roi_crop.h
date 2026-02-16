#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/normalize.h"

namespace cuframe {

struct Rect {
    int x, y, w, h;
};

// batch crop: extract num_rois regions from one NV12 frame, resize each to
// dst_w x dst_h, color convert + normalize. output is contiguous NCHW:
// output[roi * 3 * dst_h * dst_w + c * dst_h * dst_w + y * dst_w + x].
//
// rois: host-side array of bounding boxes in source pixel coordinates.
// output: device pointer with space for num_rois * 3 * dst_h * dst_w floats.
// color/norm: same as fused_nv12_to_tensor.
void roi_crop_batch(
    const uint8_t* nv12_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const Rect* rois, int num_rois,
    float* output,
    int dst_w, int dst_h,
    const ColorMatrix& color,
    const NormParams& norm,
    bool bgr = false,
    bool is_10bit = false,
    cudaStream_t stream = nullptr
);

} // namespace cuframe
