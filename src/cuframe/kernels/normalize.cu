#include "cuframe/kernels/normalize.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

NormParams make_norm_params(const float mean[3], const float std[3]) {
    NormParams p;
    for (int i = 0; i < 3; ++i) {
        p.scale[i] = 1.0f / (255.0f * std[i]);
        p.bias[i] = -mean[i] / std[i];
    }
    return p;
}

static NormParams make_imagenet_norm() {
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};
    return make_norm_params(mean, std);
}

const NormParams IMAGENET_NORM = make_imagenet_norm();

__global__ void normalize_kernel(
    const float* src, float* dst,
    int width, int height,
    float scale_r, float bias_r,
    float scale_g, float bias_g,
    float scale_b, float bias_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (x >= width || y >= height) return;

    int idx = c * width * height + y * width + x;

    float scale = (c == 0) ? scale_r : (c == 1) ? scale_g : scale_b;
    float bias  = (c == 0) ? bias_r  : (c == 1) ? bias_g  : bias_b;

    dst[idx] = src[idx] * scale + bias;
}

void normalize(
    const float* src_ptr, float* dst_ptr,
    int width, int height,
    const NormParams& params, cudaStream_t stream
) {
    dim3 block(32, 8, 1);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        3
    );
    normalize_kernel<<<grid, block, 0, stream>>>(
        src_ptr, dst_ptr, width, height,
        params.scale[0], params.bias[0],
        params.scale[1], params.bias[1],
        params.scale[2], params.bias[2]
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuframe
