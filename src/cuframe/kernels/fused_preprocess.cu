#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/nv12_sample.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

__global__ void __launch_bounds__(256, 6) fused_nv12_to_tensor_kernel(
    const uint8_t* __restrict__ nv12, float* __restrict__ rgb,
    int src_w, int src_h, unsigned int src_pitch,
    int dst_w, int dst_h,
    int pad_left, int pad_top, int inner_w, int inner_h,
    float scale_x, float scale_y, float pad_value,
    float src_offset_x, float src_offset_y,
    float3 coeff_r, float3 coeff_g, float3 coeff_b,
    float norm_scale_r, float norm_bias_r,
    float norm_scale_g, float norm_bias_g,
    float norm_scale_b, float norm_bias_b,
    bool bgr, bool is_10bit
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int plane = dst_w * dst_h;
    int pixel = y * dst_w + x;
    int r_plane = bgr ? 2 : 0;
    int b_plane = bgr ? 0 : 2;

    // padding region — write normalized pad value per channel
    int ix = x - pad_left;
    int iy = y - pad_top;
    if (ix < 0 || ix >= inner_w || iy < 0 || iy >= inner_h) {
        rgb[r_plane * plane + pixel] = pad_value * norm_scale_r + norm_bias_r;
        rgb[1       * plane + pixel] = pad_value * norm_scale_g + norm_bias_g;
        rgb[b_plane * plane + pixel] = pad_value * norm_scale_b + norm_bias_b;
        return;
    }

    // map to source coordinates (pixel center alignment + crop offset)
    float src_xf = (ix + 0.5f) * scale_x - 0.5f + src_offset_x;
    float src_yf = (iy + 0.5f) * scale_y - 0.5f + src_offset_y;

    float3 rgb_val = nv12_bilinear_sample(nv12, src_xf, src_yf,
        src_w, src_h, src_pitch, is_10bit, coeff_r, coeff_g, coeff_b);

    // normalize and write planar
    rgb[r_plane * plane + pixel] = rgb_val.x * norm_scale_r + norm_bias_r;
    rgb[1       * plane + pixel] = rgb_val.y * norm_scale_g + norm_bias_g;
    rgb[b_plane * plane + pixel] = rgb_val.z * norm_scale_b + norm_bias_b;
}

void fused_nv12_to_tensor(
    const uint8_t* nv12_ptr, float* rgb_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const ResizeParams& r, const ColorMatrix& c, const NormParams& n,
    bool bgr, bool is_10bit, cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid(
        (r.dst_w + block.x - 1) / block.x,
        (r.dst_h + block.y - 1) / block.y
    );
    fused_nv12_to_tensor_kernel<<<grid, block, 0, stream>>>(
        nv12_ptr, rgb_ptr,
        src_w, src_h, src_pitch,
        r.dst_w, r.dst_h,
        r.pad_left, r.pad_top, r.inner_w, r.inner_h,
        r.scale_x, r.scale_y, r.pad_value,
        r.src_offset_x, r.src_offset_y,
        c.r, c.g, c.b,
        n.scale[0], n.bias[0],
        n.scale[1], n.bias[1],
        n.scale[2], n.bias[2],
        bgr, is_10bit
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

void fused_preprocess_query_occupancy(int* min_grid, int* block_size) {
    cudaOccupancyMaxPotentialBlockSize(min_grid, block_size,
        fused_nv12_to_tensor_kernel, 0, 0);
}

} // namespace cuframe
