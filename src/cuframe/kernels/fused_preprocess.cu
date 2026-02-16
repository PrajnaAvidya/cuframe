#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/nv12_sample.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

struct FusedKernelParams {
    const uint8_t* __restrict__ nv12;
    float* __restrict__ rgb;
    int src_w, src_h;
    unsigned int src_pitch;
    int dst_w, dst_h;
    int pad_left, pad_top, inner_w, inner_h;
    float scale_x, scale_y, pad_value;
    float src_offset_x, src_offset_y;
    float3 coeff_r, coeff_g, coeff_b;
    float norm_scale_r, norm_bias_r;
    float norm_scale_g, norm_bias_g;
    float norm_scale_b, norm_bias_b;
    bool bgr, is_10bit;
};

__global__ void __launch_bounds__(256, 6)
fused_nv12_to_tensor_kernel(FusedKernelParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dst_w || y >= p.dst_h) return;

    int plane = p.dst_w * p.dst_h;
    int pixel = y * p.dst_w + x;
    int r_plane = p.bgr ? 2 : 0;
    int b_plane = p.bgr ? 0 : 2;

    // padding region — write normalized pad value per channel
    int ix = x - p.pad_left;
    int iy = y - p.pad_top;
    if (ix < 0 || ix >= p.inner_w || iy < 0 || iy >= p.inner_h) {
        p.rgb[r_plane * plane + pixel] = p.pad_value * p.norm_scale_r + p.norm_bias_r;
        p.rgb[1       * plane + pixel] = p.pad_value * p.norm_scale_g + p.norm_bias_g;
        p.rgb[b_plane * plane + pixel] = p.pad_value * p.norm_scale_b + p.norm_bias_b;
        return;
    }

    // map to source coordinates (pixel center alignment + crop offset)
    float src_xf = (ix + 0.5f) * p.scale_x - 0.5f + p.src_offset_x;
    float src_yf = (iy + 0.5f) * p.scale_y - 0.5f + p.src_offset_y;

    float3 rgb_val = nv12_bilinear_sample(p.nv12, src_xf, src_yf,
        p.src_w, p.src_h, p.src_pitch, p.is_10bit, p.coeff_r, p.coeff_g, p.coeff_b);

    // normalize and write planar
    p.rgb[r_plane * plane + pixel] = rgb_val.x * p.norm_scale_r + p.norm_bias_r;
    p.rgb[1       * plane + pixel] = rgb_val.y * p.norm_scale_g + p.norm_bias_g;
    p.rgb[b_plane * plane + pixel] = rgb_val.z * p.norm_scale_b + p.norm_bias_b;
}

void fused_nv12_to_tensor(
    const uint8_t* nv12_ptr, float* rgb_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const ResizeParams& r, const ColorMatrix& c, const NormParams& n,
    bool bgr, bool is_10bit, cudaStream_t stream
) {
    FusedKernelParams p{};
    p.nv12 = nv12_ptr;  p.rgb = rgb_ptr;
    p.src_w = src_w;  p.src_h = src_h;  p.src_pitch = src_pitch;
    p.dst_w = r.dst_w;  p.dst_h = r.dst_h;
    p.pad_left = r.pad_left;  p.pad_top = r.pad_top;
    p.inner_w = r.inner_w;  p.inner_h = r.inner_h;
    p.scale_x = r.scale_x;  p.scale_y = r.scale_y;  p.pad_value = r.pad_value;
    p.src_offset_x = r.src_offset_x;  p.src_offset_y = r.src_offset_y;
    p.coeff_r = c.r;  p.coeff_g = c.g;  p.coeff_b = c.b;
    p.norm_scale_r = n.scale[0];  p.norm_bias_r = n.bias[0];
    p.norm_scale_g = n.scale[1];  p.norm_bias_g = n.bias[1];
    p.norm_scale_b = n.scale[2];  p.norm_bias_b = n.bias[2];
    p.bgr = bgr;  p.is_10bit = is_10bit;

    dim3 block(32, 8);
    dim3 grid(
        (p.dst_w + block.x - 1) / block.x,
        (p.dst_h + block.y - 1) / block.y
    );
    fused_nv12_to_tensor_kernel<<<grid, block, 0, stream>>>(p);
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

void fused_preprocess_query_occupancy(int* min_grid, int* block_size) {
    cudaOccupancyMaxPotentialBlockSize(min_grid, block_size,
        fused_nv12_to_tensor_kernel, 0, 0);
}

} // namespace cuframe
