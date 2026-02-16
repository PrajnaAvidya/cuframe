#include "cuframe/kernels/roi_crop.h"
#include "cuframe/kernels/nv12_sample.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

__global__ void roi_crop_batch_kernel(
    const uint8_t* __restrict__ nv12, float* __restrict__ output,
    int src_w, int src_h, unsigned int src_pitch,
    const Rect* __restrict__ rois, int num_rois,
    int dst_w, int dst_h,
    float3 coeff_r, float3 coeff_g, float3 coeff_b,
    float norm_scale_r, float norm_bias_r,
    float norm_scale_g, float norm_bias_g,
    float norm_scale_b, float norm_bias_b,
    bool bgr, bool is_10bit
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int roi_idx = blockIdx.z;
    if (x >= dst_w || y >= dst_h || roi_idx >= num_rois) return;

    Rect roi = rois[roi_idx];
    float scale_x = (float)roi.w / dst_w;
    float scale_y = (float)roi.h / dst_h;

    // map output pixel to source coordinates within ROI
    float src_xf = (x + 0.5f) * scale_x - 0.5f + roi.x;
    float src_yf = (y + 0.5f) * scale_y - 0.5f + roi.y;

    float3 rgb_val = nv12_bilinear_sample(nv12, src_xf, src_yf,
        src_w, src_h, src_pitch, is_10bit, coeff_r, coeff_g, coeff_b);

    // write to roi's slot in output batch
    int plane = dst_w * dst_h;
    int crop_offset = roi_idx * 3 * plane;
    int pixel = y * dst_w + x;
    int r_plane = bgr ? 2 : 0;
    int b_plane = bgr ? 0 : 2;

    output[crop_offset + r_plane * plane + pixel] = rgb_val.x * norm_scale_r + norm_bias_r;
    output[crop_offset + 1       * plane + pixel] = rgb_val.y * norm_scale_g + norm_bias_g;
    output[crop_offset + b_plane * plane + pixel] = rgb_val.z * norm_scale_b + norm_bias_b;
}

void roi_crop_batch(
    const uint8_t* nv12_ptr,
    int src_w, int src_h, unsigned int src_pitch,
    const Rect* rois, int num_rois,
    float* output,
    int dst_w, int dst_h,
    const ColorMatrix& color, const NormParams& norm,
    bool bgr, bool is_10bit, cudaStream_t stream
) {
    if (num_rois == 0) return;

    // upload ROI array to device (stream-ordered alloc)
    Rect* d_rois = nullptr;
    size_t rois_bytes = num_rois * sizeof(Rect);
    CUFRAME_CUDA_CHECK(cudaMallocAsync(&d_rois, rois_bytes, stream));
    CUFRAME_CUDA_CHECK(cudaMemcpyAsync(d_rois, rois, rois_bytes,
                                        cudaMemcpyHostToDevice, stream));

    dim3 block(32, 8);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y,
        num_rois
    );
    roi_crop_batch_kernel<<<grid, block, 0, stream>>>(
        nv12_ptr, output, src_w, src_h, src_pitch,
        d_rois, num_rois, dst_w, dst_h,
        color.r, color.g, color.b,
        norm.scale[0], norm.bias[0],
        norm.scale[1], norm.bias[1],
        norm.scale[2], norm.bias[2],
        bgr, is_10bit
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
    CUFRAME_CUDA_CHECK(cudaFreeAsync(d_rois, stream));
}

} // namespace cuframe
