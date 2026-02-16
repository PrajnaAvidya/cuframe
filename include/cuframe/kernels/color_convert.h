#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuframe {

struct ColorMatrix {
    float3 r;  // {Y_coeff, U_coeff, V_coeff} for R channel
    float3 g;
    float3 b;
};

// standard matrices — defined in color_convert.cu
extern const ColorMatrix BT601;
extern const ColorMatrix BT709;

// convert NV12 pitched buffer to float32 RGB planar [0, 255].
// nv12_ptr: device pointer to NV12 data (luma then chroma, pitched layout)
// rgb_ptr: output device pointer (3 * width * height floats, planar: R, G, B)
// width, height: display dimensions (not coded/padded)
// pitch: row stride in bytes of the NV12 buffer
// matrix: color conversion coefficients
// stream: CUDA stream for async execution
void nv12_to_rgb_planar(
    const uint8_t* nv12_ptr,
    float* rgb_ptr,
    int width, int height, unsigned int pitch,
    const ColorMatrix& matrix,
    bool bgr = false,
    bool is_10bit = false,
    cudaStream_t stream = nullptr
);

void nv12_to_rgb_query_occupancy(int* min_grid, int* block_size);

} // namespace cuframe
