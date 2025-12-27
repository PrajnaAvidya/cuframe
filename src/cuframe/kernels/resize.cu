#include "cuframe/kernels/resize.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

ResizeParams make_resize_params(int src_w, int src_h, int dst_w, int dst_h) {
    ResizeParams p;
    p.src_w = src_w;
    p.src_h = src_h;
    p.dst_w = dst_w;
    p.dst_h = dst_h;
    p.pad_left = 0;
    p.pad_top = 0;
    p.inner_w = dst_w;
    p.inner_h = dst_h;
    p.scale_x = (float)src_w / (float)dst_w;
    p.scale_y = (float)src_h / (float)dst_h;
    p.pad_value = 0.0f;
    return p;
}

ResizeParams make_letterbox_params(int src_w, int src_h, int dst_w, int dst_h,
                                    float pad_value) {
    float scale_w = (float)dst_w / (float)src_w;
    float scale_h = (float)dst_h / (float)src_h;
    float scale = (scale_w < scale_h) ? scale_w : scale_h;

    int inner_w = (int)(src_w * scale + 0.5f);
    int inner_h = (int)(src_h * scale + 0.5f);

    ResizeParams p;
    p.src_w = src_w;
    p.src_h = src_h;
    p.dst_w = dst_w;
    p.dst_h = dst_h;
    p.pad_left = (dst_w - inner_w) / 2;
    p.pad_top = (dst_h - inner_h) / 2;
    p.inner_w = inner_w;
    p.inner_h = inner_h;
    p.scale_x = (float)src_w / (float)inner_w;
    p.scale_y = (float)src_h / (float)inner_h;
    p.pad_value = pad_value;
    return p;
}

__global__ void resize_bilinear_kernel(
    const float* src, float* dst,
    int src_w, int src_h, int dst_w, int dst_h,
    int pad_left, int pad_top, int inner_w, int inner_h,
    float scale_x, float scale_y, float pad_value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (x >= dst_w || y >= dst_h) return;

    int dst_idx = c * dst_w * dst_h + y * dst_w + x;

    // check padding region
    int ix = x - pad_left;
    int iy = y - pad_top;
    if (ix < 0 || ix >= inner_w || iy < 0 || iy >= inner_h) {
        dst[dst_idx] = pad_value;
        return;
    }

    // map to source coordinates (pixel center alignment)
    float src_x = (ix + 0.5f) * scale_x - 0.5f;
    float src_y = (iy + 0.5f) * scale_y - 0.5f;

    // bilinear sample
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    float fx = src_x - x0;
    float fy = src_y - y0;

    // clamp to source bounds
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    const float* plane = src + c * src_w * src_h;
    float v00 = plane[y0 * src_w + x0];
    float v01 = plane[y0 * src_w + x1];
    float v10 = plane[y1 * src_w + x0];
    float v11 = plane[y1 * src_w + x1];

    dst[dst_idx] = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01
                 + (1 - fx) * fy * v10       + fx * fy * v11;
}

void resize_bilinear(
    const float* src_ptr, float* dst_ptr,
    const ResizeParams& p, cudaStream_t stream
) {
    dim3 block(32, 8, 1);
    dim3 grid(
        (p.dst_w + block.x - 1) / block.x,
        (p.dst_h + block.y - 1) / block.y,
        3
    );
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        src_ptr, dst_ptr,
        p.src_w, p.src_h, p.dst_w, p.dst_h,
        p.pad_left, p.pad_top, p.inner_w, p.inner_h,
        p.scale_x, p.scale_y, p.pad_value
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuframe
