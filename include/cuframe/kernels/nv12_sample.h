#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuframe {

// read one NV12/P016 pixel, apply color matrix, return clamped RGB [0, 255].
// is_10bit: P016 has 16-bit MSB-aligned samples, >> 8 maps to [0, 255].
// branch is uniform (all threads same value) so zero divergence cost.
__device__ inline float3 nv12_to_rgb(
    const uint8_t* __restrict__ frame, int x, int y,
    int src_h, unsigned int pitch, bool is_10bit,
    float3 coeff_r, float3 coeff_g, float3 coeff_b
) {
    float Y, U, V;
    if (is_10bit) {
        auto* y_row = (const uint16_t*)(frame + y * pitch);
        Y = (float)(y_row[x] >> 8);
        auto* uv_row = (const uint16_t*)(frame + pitch * src_h + (y / 2) * pitch);
        U = (float)(uv_row[(x / 2) * 2] >> 8) - 128.0f;
        V = (float)(uv_row[(x / 2) * 2 + 1] >> 8) - 128.0f;
    } else {
        Y = (float)frame[y * pitch + x];
        int chroma_off = pitch * src_h + (y / 2) * pitch + (x / 2) * 2;
        U = (float)frame[chroma_off] - 128.0f;
        V = (float)frame[chroma_off + 1] - 128.0f;
    }

    return make_float3(
        fminf(fmaxf(Y * coeff_r.x + U * coeff_r.y + V * coeff_r.z, 0.0f), 255.0f),
        fminf(fmaxf(Y * coeff_g.x + U * coeff_g.y + V * coeff_g.z, 0.0f), 255.0f),
        fminf(fmaxf(Y * coeff_b.x + U * coeff_b.y + V * coeff_b.z, 0.0f), 255.0f)
    );
}

// bilinear sample 4 NV12 points and interpolate to RGB
__device__ inline float3 nv12_bilinear_sample(
    const uint8_t* __restrict__ frame, float src_xf, float src_yf,
    int src_w, int src_h, unsigned int pitch, bool is_10bit,
    float3 coeff_r, float3 coeff_g, float3 coeff_b
) {
    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    float fx = src_xf - x0;
    float fy = src_yf - y0;

    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float3 c00 = nv12_to_rgb(frame, x0, y0, src_h, pitch, is_10bit, coeff_r, coeff_g, coeff_b);
    float3 c01 = nv12_to_rgb(frame, x1, y0, src_h, pitch, is_10bit, coeff_r, coeff_g, coeff_b);
    float3 c10 = nv12_to_rgb(frame, x0, y1, src_h, pitch, is_10bit, coeff_r, coeff_g, coeff_b);
    float3 c11 = nv12_to_rgb(frame, x1, y1, src_h, pitch, is_10bit, coeff_r, coeff_g, coeff_b);

    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    return make_float3(
        w00 * c00.x + w01 * c01.x + w10 * c10.x + w11 * c11.x,
        w00 * c00.y + w01 * c01.y + w10 * c10.y + w11 * c11.y,
        w00 * c00.z + w01 * c01.z + w10 * c10.z + w11 * c11.z
    );
}

} // namespace cuframe
