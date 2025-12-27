#include "cuframe/kernels/resize.h"
#include "cuframe/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// helper: allocate device buffer, copy host data up, return device pointer.
// caller must cudaFree.
static float* upload(const std::vector<float>& host) {
    float* d = nullptr;
    size_t bytes = host.size() * sizeof(float);
    CUFRAME_CUDA_CHECK(cudaMalloc(&d, bytes));
    CUFRAME_CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

// helper: download device buffer to host vector
static std::vector<float> download(const float* d, size_t count) {
    std::vector<float> host(count);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), d, count * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

// generate a horizontal gradient: pixel value = (float)x for each channel
static std::vector<float> make_gradient(int w, int h) {
    std::vector<float> img(3 * w * h);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                img[c * w * h + y * w + x] = (float)x;
            }
        }
    }
    return img;
}

// generate a solid color image
static std::vector<float> make_solid(int w, int h, float r, float g, float b) {
    std::vector<float> img(3 * w * h);
    int plane = w * h;
    for (int i = 0; i < plane; ++i) {
        img[0 * plane + i] = r;
        img[1 * plane + i] = g;
        img[2 * plane + i] = b;
    }
    return img;
}

TEST(ResizeTest, IdentityResize) {
    const int w = 64, h = 48;
    auto src_host = make_gradient(w, h);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    auto params = cuframe::make_resize_params(w, h, w, h);
    cuframe::resize_bilinear(src_d, dst_d, params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * w * h);

    for (size_t i = 0; i < src_host.size(); ++i) {
        EXPECT_NEAR(dst_host[i], src_host[i], 0.01f)
            << "mismatch at index " << i;
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(ResizeTest, DownscaleHalf) {
    const int src_w = 64, src_h = 48;
    const int dst_w = 32, dst_h = 24;

    auto src_host = make_gradient(src_w, src_h);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * dst_w * dst_h * sizeof(float)));

    auto params = cuframe::make_resize_params(src_w, src_h, dst_w, dst_h);
    cuframe::resize_bilinear(src_d, dst_d, params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * dst_w * dst_h);

    // gradient image: value = x in source. after 2x downscale with pixel center
    // alignment, output pixel x maps to source (x+0.5)*2 - 0.5 = 2x + 0.5.
    // bilinear at src_x=2x+0.5 interpolates between floor(2x+0.5) and ceil(2x+0.5).
    // for x=0: src_x=0.5, floor=0, ceil=1, fx=0.5 → val = 0.5*0 + 0.5*1 = 0.5
    // for x=1: src_x=2.5, floor=2, ceil=3, fx=0.5 → val = 0.5*2 + 0.5*3 = 2.5
    // general: expected = 2*x + 0.5
    int plane = dst_w * dst_h;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_h; ++y) {
            float val = dst_host[c * plane + y * dst_w + 0];
            EXPECT_NEAR(val, 0.5f, 0.01f)
                << "first pixel c=" << c << " y=" << y;

            val = dst_host[c * plane + y * dst_w + 1];
            EXPECT_NEAR(val, 2.5f, 0.01f)
                << "second pixel c=" << c << " y=" << y;
        }
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(ResizeTest, UpscaleDouble) {
    const int src_w = 32, src_h = 24;
    const int dst_w = 64, dst_h = 48;

    auto src_host = make_gradient(src_w, src_h);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * dst_w * dst_h * sizeof(float)));

    auto params = cuframe::make_resize_params(src_w, src_h, dst_w, dst_h);
    cuframe::resize_bilinear(src_d, dst_d, params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * dst_w * dst_h);

    // verify not all zeros and values are reasonable
    int plane = dst_w * dst_h;
    bool has_nonzero = false;
    for (size_t i = 0; i < dst_host.size(); ++i) {
        if (dst_host[i] != 0.0f) has_nonzero = true;
        EXPECT_GE(dst_host[i], 0.0f);
        EXPECT_LE(dst_host[i], (float)(src_w - 1) + 0.01f);
    }
    EXPECT_TRUE(has_nonzero);

    // upscale from gradient: output pixel x=0 maps to src_x = (0+0.5)*0.5 - 0.5 = -0.25
    // clamped to 0 → val = 0.0. pixel x=1 maps to src_x = 0.25 → val = 0.25.
    float v0 = dst_host[0 * plane + 0 * dst_w + 0];
    EXPECT_NEAR(v0, 0.0f, 0.01f);

    float v1 = dst_host[0 * plane + 0 * dst_w + 1];
    EXPECT_NEAR(v1, 0.25f, 0.01f);

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(ResizeTest, LetterboxAspectRatio) {
    // 320x240 (4:3) → 640x640 (1:1)
    // scale = min(640/320, 640/240) = min(2.0, 2.667) = 2.0
    // inner_w = 640, inner_h = 480
    // pad_left = 0, pad_top = (640-480)/2 = 80
    const int src_w = 320, src_h = 240;
    const int dst_w = 640, dst_h = 640;

    auto src_host = make_solid(src_w, src_h, 100.0f, 150.0f, 200.0f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * dst_w * dst_h * sizeof(float)));

    auto params = cuframe::make_letterbox_params(src_w, src_h, dst_w, dst_h);
    cuframe::resize_bilinear(src_d, dst_d, params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * dst_w * dst_h);
    int plane = dst_w * dst_h;

    // top padding region (y < 80): should be 114.0f
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 80; ++y) {
            for (int x = 0; x < dst_w; x += 37) {
                float val = dst_host[c * plane + y * dst_w + x];
                EXPECT_FLOAT_EQ(val, 114.0f)
                    << "padding at c=" << c << " x=" << x << " y=" << y;
            }
        }
    }

    // bottom padding region (y >= 560): should be 114.0f
    for (int c = 0; c < 3; ++c) {
        for (int y = 560; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; x += 37) {
                float val = dst_host[c * plane + y * dst_w + x];
                EXPECT_FLOAT_EQ(val, 114.0f)
                    << "padding at c=" << c << " x=" << x << " y=" << y;
            }
        }
    }

    // image region center pixel (y=320, x=320): should be source color
    float r = dst_host[0 * plane + 320 * dst_w + 320];
    float g = dst_host[1 * plane + 320 * dst_w + 320];
    float b = dst_host[2 * plane + 320 * dst_w + 320];
    EXPECT_NEAR(r, 100.0f, 0.5f);
    EXPECT_NEAR(g, 150.0f, 0.5f);
    EXPECT_NEAR(b, 200.0f, 0.5f);

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(ResizeTest, LetterboxCustomPadValue) {
    const int src_w = 320, src_h = 240;
    const int dst_w = 640, dst_h = 640;
    const float custom_pad = 0.0f;

    auto src_host = make_solid(src_w, src_h, 100.0f, 150.0f, 200.0f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * dst_w * dst_h * sizeof(float)));

    auto params = cuframe::make_letterbox_params(src_w, src_h, dst_w, dst_h, custom_pad);
    cuframe::resize_bilinear(src_d, dst_d, params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * dst_w * dst_h);
    int plane = dst_w * dst_h;

    // top padding should be 0.0f, not 114.0f
    for (int c = 0; c < 3; ++c) {
        float val = dst_host[c * plane + 0 * dst_w + 0];
        EXPECT_FLOAT_EQ(val, custom_pad)
            << "custom pad value not respected, c=" << c;
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(ResizeTest, MakeResizeParamsCorrect) {
    auto p = cuframe::make_resize_params(320, 240, 640, 480);

    EXPECT_EQ(p.src_w, 320);
    EXPECT_EQ(p.src_h, 240);
    EXPECT_EQ(p.dst_w, 640);
    EXPECT_EQ(p.dst_h, 480);
    EXPECT_EQ(p.pad_left, 0);
    EXPECT_EQ(p.pad_top, 0);
    EXPECT_EQ(p.inner_w, 640);
    EXPECT_EQ(p.inner_h, 480);
    EXPECT_FLOAT_EQ(p.scale_x, 320.0f / 640.0f);
    EXPECT_FLOAT_EQ(p.scale_y, 240.0f / 480.0f);
}

TEST(ResizeTest, MakeLetterboxParamsCorrect) {
    auto p = cuframe::make_letterbox_params(320, 240, 640, 640);

    // scale = min(640/320, 640/240) = 2.0
    // inner_w = 640, inner_h = 480
    EXPECT_EQ(p.src_w, 320);
    EXPECT_EQ(p.src_h, 240);
    EXPECT_EQ(p.dst_w, 640);
    EXPECT_EQ(p.dst_h, 640);
    EXPECT_EQ(p.inner_w, 640);
    EXPECT_EQ(p.inner_h, 480);
    EXPECT_EQ(p.pad_left, 0);
    EXPECT_EQ(p.pad_top, 80);
    EXPECT_FLOAT_EQ(p.scale_x, 320.0f / 640.0f);
    EXPECT_FLOAT_EQ(p.scale_y, 240.0f / 480.0f);
    EXPECT_FLOAT_EQ(p.pad_value, 114.0f);
}

TEST(ResizeTest, StreamExecution) {
    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    const int w = 32, h = 24;
    auto src_host = make_gradient(w, h);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    auto params = cuframe::make_resize_params(w, h, w, h);
    cuframe::resize_bilinear(src_d, dst_d, params, stream);
    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto dst_host = download(dst_d, 3 * w * h);

    // basic sanity: not all zeros
    bool has_nonzero = false;
    for (auto v : dst_host) {
        if (v != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);

    cudaFree(src_d);
    cudaFree(dst_d);
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));
}
