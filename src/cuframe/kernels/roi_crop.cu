#include "cuframe/kernels/roi_crop.h"
#include "cuframe/kernels/nv12_sample.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

struct RoiCropParams {
    const uint8_t* __restrict__ nv12;
    float* __restrict__ output;
    int src_w, src_h;
    unsigned int src_pitch;
    const Rect* __restrict__ rois;
    int num_rois;
    int dst_w, dst_h;
    float3 coeff_r, coeff_g, coeff_b;
    float norm_scale_r, norm_bias_r;
    float norm_scale_g, norm_bias_g;
    float norm_scale_b, norm_bias_b;
    bool bgr, is_10bit;
};

__global__ void roi_crop_batch_kernel(RoiCropParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int roi_idx = blockIdx.z;
    if (x >= p.dst_w || y >= p.dst_h || roi_idx >= p.num_rois) return;

    Rect roi = p.rois[roi_idx];
    float scale_x = (float)roi.w / p.dst_w;
    float scale_y = (float)roi.h / p.dst_h;

    // map output pixel to source coordinates within ROI
    float src_xf = (x + 0.5f) * scale_x - 0.5f + roi.x;
    float src_yf = (y + 0.5f) * scale_y - 0.5f + roi.y;

    float3 rgb_val = nv12_bilinear_sample(p.nv12, src_xf, src_yf,
        p.src_w, p.src_h, p.src_pitch, p.is_10bit, p.coeff_r, p.coeff_g, p.coeff_b);

    // write to roi's slot in output batch
    int plane = p.dst_w * p.dst_h;
    int crop_offset = roi_idx * 3 * plane;
    int pixel = y * p.dst_w + x;
    int r_plane = p.bgr ? 2 : 0;
    int b_plane = p.bgr ? 0 : 2;

    p.output[crop_offset + r_plane * plane + pixel] = rgb_val.x * p.norm_scale_r + p.norm_bias_r;
    p.output[crop_offset + 1       * plane + pixel] = rgb_val.y * p.norm_scale_g + p.norm_bias_g;
    p.output[crop_offset + b_plane * plane + pixel] = rgb_val.z * p.norm_scale_b + p.norm_bias_b;
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

    RoiCropParams p{};
    p.nv12 = nv12_ptr;  p.output = output;
    p.src_w = src_w;  p.src_h = src_h;  p.src_pitch = src_pitch;
    p.rois = d_rois;  p.num_rois = num_rois;
    p.dst_w = dst_w;  p.dst_h = dst_h;
    p.coeff_r = color.r;  p.coeff_g = color.g;  p.coeff_b = color.b;
    p.norm_scale_r = norm.scale[0];  p.norm_bias_r = norm.bias[0];
    p.norm_scale_g = norm.scale[1];  p.norm_bias_g = norm.bias[1];
    p.norm_scale_b = norm.scale[2];  p.norm_bias_b = norm.bias[2];
    p.bgr = bgr;  p.is_10bit = is_10bit;

    dim3 block(32, 8);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y,
        num_rois
    );
    roi_crop_batch_kernel<<<grid, block, 0, stream>>>(p);
    CUFRAME_CUDA_CHECK(cudaGetLastError());
    CUFRAME_CUDA_CHECK(cudaFreeAsync(d_rois, stream));
}

} // namespace cuframe
