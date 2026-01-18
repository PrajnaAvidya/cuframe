#pragma once
#include <cuda_runtime.h>

namespace cuframe {

struct ResizeParams {
    int src_w;
    int src_h;
    int dst_w;           // total output width (including padding)
    int dst_h;           // total output height (including padding)
    int pad_left;        // padding pixels on left
    int pad_top;         // padding pixels on top
    int inner_w;         // scaled image width
    int inner_h;         // scaled image height
    float scale_x;      // src_w / inner_w (source pixels per output pixel)
    float scale_y;      // src_h / inner_h
    float pad_value;    // fill value for padding region (default 114.0f)
    float src_offset_x = 0.0f;  // crop offset in source pixels
    float src_offset_y = 0.0f;
};

// plain (stretch) resize — no padding, inner == dst
ResizeParams make_resize_params(int src_w, int src_h, int dst_w, int dst_h);

// letterbox — preserve aspect ratio, center, pad borders
ResizeParams make_letterbox_params(int src_w, int src_h, int dst_w, int dst_h,
                                    float pad_value = 114.0f);

// center crop: compute resize params that map the center crop_w × crop_h region
// of a (conceptually) resized image back to source coordinates.
// if resize_w/resize_h are 0, crop directly from source at source resolution.
ResizeParams make_center_crop_params(int src_w, int src_h,
                                      int resize_w, int resize_h,
                                      int crop_w, int crop_h);

// resize float32 RGB planar (3 × src_h × src_w) → (3 × dst_h × dst_w).
// bilinear interpolation in image region, pad_value in border region.
void resize_bilinear(
    const float* src_ptr,   // 3 × src_h × src_w
    float* dst_ptr,         // 3 × dst_h × dst_w
    const ResizeParams& params,
    cudaStream_t stream = nullptr
);

} // namespace cuframe
