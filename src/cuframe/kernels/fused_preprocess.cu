#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

// read one NV12 pixel, apply color matrix, return clamped RGB
__device__ inline float3 nv12_to_rgb(
    const uint8_t* nv12, int x, int y,
    int src_h, unsigned int pitch,
    float3 coeff_r, float3 coeff_g, float3 coeff_b
) {
    float Y = (float)nv12[y * pitch + x];
    int chroma_off = pitch * src_h + (y / 2) * pitch + (x / 2) * 2;
    float U = (float)nv12[chroma_off] - 128.0f;
    float V = (float)nv12[chroma_off + 1] - 128.0f;

    return make_float3(
        fminf(fmaxf(Y * coeff_r.x + U * coeff_r.y + V * coeff_r.z, 0.0f), 255.0f),
        fminf(fmaxf(Y * coeff_g.x + U * coeff_g.y + V * coeff_g.z, 0.0f), 255.0f),
        fminf(fmaxf(Y * coeff_b.x + U * coeff_b.y + V * coeff_b.z, 0.0f), 255.0f)
    );
}

__global__ void fused_nv12_to_tensor_kernel(
    const uint8_t* nv12, float* rgb,
    int src_w, int src_h, unsigned int src_pitch,
    int dst_w, int dst_h,
    int pad_left, int pad_top, int inner_w, int inner_h,
    float scale_x, float scale_y, float pad_value,
    float3 coeff_r, float3 coeff_g, float3 coeff_b,
    float norm_scale_r, float norm_bias_r,
    float norm_scale_g, float norm_bias_g,
    float norm_scale_b, float norm_bias_b,
    bool bgr
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

    // map to source coordinates (pixel center alignment)
    float src_xf = (ix + 0.5f) * scale_x - 0.5f;
    float src_yf = (iy + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    float fx = src_xf - x0;
    float fy = src_yf - y0;

    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    // color convert 4 NV12 sample points to RGB
    float3 c00 = nv12_to_rgb(nv12, x0, y0, src_h, src_pitch, coeff_r, coeff_g, coeff_b);
    float3 c01 = nv12_to_rgb(nv12, x1, y0, src_h, src_pitch, coeff_r, coeff_g, coeff_b);
    float3 c10 = nv12_to_rgb(nv12, x0, y1, src_h, src_pitch, coeff_r, coeff_g, coeff_b);
    float3 c11 = nv12_to_rgb(nv12, x1, y1, src_h, src_pitch, coeff_r, coeff_g, coeff_b);

    // bilinear interpolation in RGB space
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    float R = w00 * c00.x + w01 * c01.x + w10 * c10.x + w11 * c11.x;
    float G = w00 * c00.y + w01 * c01.y + w10 * c10.y + w11 * c11.y;
    float B = w00 * c00.z + w01 * c01.z + w10 * c10.z + w11 * c11.z;

    // normalize and write planar
    rgb[r_plane * plane + pixel] = R * norm_scale_r + norm_bias_r;
    rgb[1       * plane + pixel] = G * norm_scale_g + norm_bias_g;
    rgb[b_plane * plane + pixel] = B * norm_scale_b + norm_bias_b;
}

void fused_nv12_to_tensor(
    const uint8_t* nv12_ptr, float* rgb_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const ResizeParams& r, const ColorMatrix& c, const NormParams& n,
    bool bgr, cudaStream_t stream
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
        c.r, c.g, c.b,
        n.scale[0], n.bias[0],
        n.scale[1], n.bias[1],
        n.scale[2], n.bias[2],
        bgr
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuframe
