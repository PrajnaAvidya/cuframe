#include "cuframe/kernels/normalize.h"
#include "cuframe/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// helper: allocate device buffer, copy host data up
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

// generate solid-value planar image (3 × h × w), same value all channels
static std::vector<float> make_solid(int w, int h, float val) {
    return std::vector<float>(3 * w * h, val);
}

TEST(NormalizeTest, KnownValues) {
    const int w = 4, h = 4;
    auto src_host = make_solid(w, h, 127.5f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    cuframe::normalize(src_d, dst_d, w, h, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * w * h);
    int plane = w * h;

    // expected: 127.5 * scale[c] + bias[c]
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        float scale = 1.0f / (255.0f * std[c]);
        float bias = -mean[c] / std[c];
        float expected = 127.5f * scale + bias;
        for (int i = 0; i < plane; ++i) {
            EXPECT_NEAR(dst_host[c * plane + i], expected, 1e-5f)
                << "c=" << c << " i=" << i;
        }
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(NormalizeTest, ZeroInput) {
    const int w = 4, h = 4;
    auto src_host = make_solid(w, h, 0.0f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    cuframe::normalize(src_d, dst_d, w, h, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * w * h);
    int plane = w * h;

    // output should be bias[c] = -mean[c]/std[c]
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        float expected = -mean[c] / std[c];
        for (int i = 0; i < plane; ++i) {
            EXPECT_NEAR(dst_host[c * plane + i], expected, 1e-5f)
                << "c=" << c << " i=" << i;
        }
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(NormalizeTest, MaxInput) {
    const int w = 4, h = 4;
    auto src_host = make_solid(w, h, 255.0f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    cuframe::normalize(src_d, dst_d, w, h, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto dst_host = download(dst_d, 3 * w * h);
    int plane = w * h;

    // output should be (1.0 - mean[c]) / std[c]
    // which equals 255 * scale[c] + bias[c]
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        float expected = (1.0f - mean[c]) / std[c];
        for (int i = 0; i < plane; ++i) {
            EXPECT_NEAR(dst_host[c * plane + i], expected, 1e-4f)
                << "c=" << c << " i=" << i;
        }
    }

    cudaFree(src_d);
    cudaFree(dst_d);
}

TEST(NormalizeTest, InPlace) {
    const int w = 4, h = 4;
    auto src_host = make_solid(w, h, 127.5f);
    float* d = upload(src_host);

    // same pointer for src and dst
    cuframe::normalize(d, d, w, h, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto result = download(d, 3 * w * h);
    int plane = w * h;

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        float scale = 1.0f / (255.0f * std[c]);
        float bias = -mean[c] / std[c];
        float expected = 127.5f * scale + bias;
        EXPECT_NEAR(result[c * plane], expected, 1e-5f) << "c=" << c;
    }

    cudaFree(d);
}

TEST(NormalizeTest, CustomParams) {
    const int w = 4, h = 4;
    auto src_host = make_solid(w, h, 127.5f);
    float* src_d = upload(src_host);

    float* dst_imagenet = nullptr;
    float* dst_custom = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_imagenet, 3 * w * h * sizeof(float)));
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_custom, 3 * w * h * sizeof(float)));

    // run with ImageNet params
    cuframe::normalize(src_d, dst_imagenet, w, h, cuframe::IMAGENET_NORM);

    // run with custom params
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float std[]  = {0.5f, 0.5f, 0.5f};
    auto custom = cuframe::make_norm_params(mean, std);
    cuframe::normalize(src_d, dst_custom, w, h, custom);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto host_imagenet = download(dst_imagenet, 3 * w * h);
    auto host_custom = download(dst_custom, 3 * w * h);

    // different params → different output
    int diff_count = 0;
    for (size_t i = 0; i < host_imagenet.size(); ++i) {
        if (std::abs(host_imagenet[i] - host_custom[i]) > 1e-5f) ++diff_count;
    }
    EXPECT_GT(diff_count, 0) << "custom params produced identical output to ImageNet";

    // verify custom output: 127.5 / (255 * 0.5) + (-0.5 / 0.5) = 1.0 - 1.0 = 0.0
    EXPECT_NEAR(host_custom[0], 0.0f, 1e-5f);

    cudaFree(src_d);
    cudaFree(dst_imagenet);
    cudaFree(dst_custom);
}

TEST(NormalizeTest, StreamExecution) {
    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    const int w = 8, h = 8;
    auto src_host = make_solid(w, h, 100.0f);
    float* src_d = upload(src_host);

    float* dst_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&dst_d, 3 * w * h * sizeof(float)));

    cuframe::normalize(src_d, dst_d, w, h, cuframe::IMAGENET_NORM, stream);
    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto dst_host = download(dst_d, 3 * w * h);

    // basic sanity: values should be in typical normalized range
    for (size_t i = 0; i < dst_host.size(); ++i) {
        EXPECT_GT(dst_host[i], -3.0f);
        EXPECT_LT(dst_host[i], 3.0f);
    }

    cudaFree(src_d);
    cudaFree(dst_d);
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));
}
