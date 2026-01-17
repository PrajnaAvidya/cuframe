#include "cuframe/kernels/color_convert.h"
#include "cuframe/cuda_utils.h"

namespace cuframe {

const ColorMatrix BT601 = {
    {1.0f,  0.0f,      1.402f   },
    {1.0f, -0.344136f, -0.714136f},
    {1.0f,  1.772f,     0.0f     }
};

const ColorMatrix BT709 = {
    {1.0f,  0.0f,      1.5748f },
    {1.0f, -0.1873f,  -0.4681f },
    {1.0f,  1.8556f,    0.0f   }
};

__global__ void nv12_to_rgb_planar_kernel(
    const uint8_t* nv12, float* rgb,
    int width, int height, unsigned int pitch,
    float3 coeff_r, float3 coeff_g, float3 coeff_b,
    bool bgr
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Y from luma plane
    float Y = (float)nv12[y * pitch + x];

    // UV from chroma plane (starts at pitch * height, interleaved UVUV)
    int chroma_offset = pitch * height;
    int uv_x = (x / 2) * 2;
    float U = (float)nv12[chroma_offset + (y / 2) * pitch + uv_x] - 128.0f;
    float V = (float)nv12[chroma_offset + (y / 2) * pitch + uv_x + 1] - 128.0f;

    // apply color matrix, clamp to [0, 255]
    float R = fminf(fmaxf(Y * coeff_r.x + U * coeff_r.y + V * coeff_r.z, 0.0f), 255.0f);
    float G = fminf(fmaxf(Y * coeff_g.x + U * coeff_g.y + V * coeff_g.z, 0.0f), 255.0f);
    float B = fminf(fmaxf(Y * coeff_b.x + U * coeff_b.y + V * coeff_b.z, 0.0f), 255.0f);

    // write planar: R/B plane, G plane, B/R plane contiguous
    int pixel = y * width + x;
    int plane = width * height;
    int r_plane = bgr ? 2 : 0;
    int b_plane = bgr ? 0 : 2;
    rgb[r_plane * plane + pixel] = R;
    rgb[1       * plane + pixel] = G;
    rgb[b_plane * plane + pixel] = B;
}

void nv12_to_rgb_planar(
    const uint8_t* nv12_ptr, float* rgb_ptr,
    int width, int height, unsigned int pitch,
    const ColorMatrix& matrix, bool bgr, cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nv12_to_rgb_planar_kernel<<<grid, block, 0, stream>>>(
        nv12_ptr, rgb_ptr, width, height, pitch,
        matrix.r, matrix.g, matrix.b,
        bgr
    );
    CUFRAME_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuframe
