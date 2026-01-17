#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/cuda_utils.h"
#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture: decode first frame, provide NV12 data for both fused and separate paths
class FusedPreprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found";
        demuxer_ = std::make_unique<cuframe::Demuxer>(TEST_VIDEO);
        decoder_ = std::make_unique<cuframe::Decoder>(demuxer_->video_info(), 10);
        decode_first_frame();

        w_ = frames_[0].width;
        h_ = frames_[0].height;
        pitch_ = frames_[0].pitch;
        nv12_ptr_ = static_cast<const uint8_t*>(frames_[0].buffer->data());
    }

    void decode_first_frame() {
        AVPacket* pkt = av_packet_alloc();
        while (demuxer_->read_packet(pkt)) {
            decoder_->decode(pkt, frames_);
            av_packet_unref(pkt);
            if (!frames_.empty()) break;
        }
        if (frames_.empty()) decoder_->flush(frames_);
        av_packet_free(&pkt);
        ASSERT_FALSE(frames_.empty()) << "failed to decode any frames";
    }

    // run the separate pipeline: color convert → resize → normalize
    std::vector<float> run_separate(int dst_w, int dst_h,
                                     const cuframe::ResizeParams& resize_params,
                                     const cuframe::ColorMatrix& color,
                                     const cuframe::NormParams& norm) {
        size_t rgb_size = 3 * w_ * h_ * sizeof(float);
        size_t resized_size = 3 * dst_w * dst_h * sizeof(float);

        float* rgb_d = nullptr;
        float* resized_d = nullptr;
        float* normed_d = nullptr;
        CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_d, rgb_size));
        CUFRAME_CUDA_CHECK(cudaMalloc(&resized_d, resized_size));
        CUFRAME_CUDA_CHECK(cudaMalloc(&normed_d, resized_size));

        cuframe::nv12_to_rgb_planar(nv12_ptr_, rgb_d, w_, h_, pitch_, color);
        cuframe::resize_bilinear(rgb_d, resized_d, resize_params);
        cuframe::normalize(resized_d, normed_d, dst_w, dst_h, norm);
        CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> host(3 * dst_w * dst_h);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), normed_d,
                                       resized_size, cudaMemcpyDeviceToHost));

        cudaFree(rgb_d);
        cudaFree(resized_d);
        cudaFree(normed_d);
        return host;
    }

    // run the fused pipeline
    std::vector<float> run_fused(int dst_w, int dst_h,
                                  const cuframe::ResizeParams& resize_params,
                                  const cuframe::ColorMatrix& color,
                                  const cuframe::NormParams& norm) {
        size_t out_size = 3 * dst_w * dst_h * sizeof(float);
        float* fused_d = nullptr;
        CUFRAME_CUDA_CHECK(cudaMalloc(&fused_d, out_size));

        cuframe::fused_nv12_to_tensor(nv12_ptr_, fused_d, w_, h_, pitch_,
                                       resize_params, color, norm);
        CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> host(3 * dst_w * dst_h);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), fused_d,
                                       out_size, cudaMemcpyDeviceToHost));
        cudaFree(fused_d);
        return host;
    }

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;

    const uint8_t* nv12_ptr_ = nullptr;
    int w_ = 0, h_ = 0;
    unsigned int pitch_ = 0;
};

TEST_F(FusedPreprocessTest, EquivalenceWithSeparatePipeline) {
    const int dst_w = 640, dst_h = 640;
    auto params = cuframe::make_letterbox_params(w_, h_, dst_w, dst_h);

    auto separate = run_separate(dst_w, dst_h, params, cuframe::BT601, cuframe::IMAGENET_NORM);
    auto fused = run_fused(dst_w, dst_h, params, cuframe::BT601, cuframe::IMAGENET_NORM);

    ASSERT_EQ(separate.size(), fused.size());

    int mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < separate.size(); ++i) {
        float diff = std::abs(separate[i] - fused[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) ++mismatch_count;
    }

    EXPECT_EQ(mismatch_count, 0)
        << mismatch_count << " mismatches out of " << separate.size()
        << ", max diff = " << max_diff;
}

TEST_F(FusedPreprocessTest, LetterboxPadding) {
    const int dst_w = 640, dst_h = 640;
    auto params = cuframe::make_letterbox_params(w_, h_, dst_w, dst_h);
    auto fused = run_fused(dst_w, dst_h, params, cuframe::BT601, cuframe::IMAGENET_NORM);

    int plane = dst_w * dst_h;
    int pad_top = params.pad_top;

    // compute expected normalized pad value per channel
    const float pad = params.pad_value;
    float expected_r = pad * cuframe::IMAGENET_NORM.scale[0] + cuframe::IMAGENET_NORM.bias[0];
    float expected_g = pad * cuframe::IMAGENET_NORM.scale[1] + cuframe::IMAGENET_NORM.bias[1];
    float expected_b = pad * cuframe::IMAGENET_NORM.scale[2] + cuframe::IMAGENET_NORM.bias[2];

    // top padding (y < pad_top)
    for (int y = 0; y < pad_top; ++y) {
        for (int x = 0; x < dst_w; x += 37) {
            int pixel = y * dst_w + x;
            EXPECT_NEAR(fused[0 * plane + pixel], expected_r, 1e-5f)
                << "R pad at x=" << x << " y=" << y;
            EXPECT_NEAR(fused[1 * plane + pixel], expected_g, 1e-5f)
                << "G pad at x=" << x << " y=" << y;
            EXPECT_NEAR(fused[2 * plane + pixel], expected_b, 1e-5f)
                << "B pad at x=" << x << " y=" << y;
        }
    }

    // bottom padding (y >= pad_top + inner_h)
    int bottom_start = pad_top + params.inner_h;
    for (int y = bottom_start; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; x += 37) {
            int pixel = y * dst_w + x;
            EXPECT_NEAR(fused[0 * plane + pixel], expected_r, 1e-5f)
                << "R pad at x=" << x << " y=" << y;
        }
    }

    // image region center should NOT be pad value
    int cy = pad_top + params.inner_h / 2;
    int cx = dst_w / 2;
    int center_pixel = cy * dst_w + cx;
    bool is_pad = std::abs(fused[0 * plane + center_pixel] - expected_r) < 1e-5f
               && std::abs(fused[1 * plane + center_pixel] - expected_g) < 1e-5f
               && std::abs(fused[2 * plane + center_pixel] - expected_b) < 1e-5f;
    EXPECT_FALSE(is_pad) << "center of image region should not be pad value";
}

TEST_F(FusedPreprocessTest, IdentitySizeNoLetterbox) {
    // same resolution, no padding — compare fused vs color convert + normalize (identity resize)
    auto resize_params = cuframe::make_resize_params(w_, h_, w_, h_);

    auto separate = run_separate(w_, h_, resize_params, cuframe::BT601, cuframe::IMAGENET_NORM);
    auto fused = run_fused(w_, h_, resize_params, cuframe::BT601, cuframe::IMAGENET_NORM);

    ASSERT_EQ(separate.size(), fused.size());

    int mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < separate.size(); ++i) {
        float diff = std::abs(separate[i] - fused[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) ++mismatch_count;
    }

    EXPECT_EQ(mismatch_count, 0)
        << mismatch_count << " mismatches, max diff = " << max_diff;
}

TEST_F(FusedPreprocessTest, BGRSwapsChannels) {
    const int dst_w = 640, dst_h = 640;
    auto params = cuframe::make_letterbox_params(w_, h_, dst_w, dst_h);
    size_t out_size = 3 * dst_w * dst_h * sizeof(float);
    int plane = dst_w * dst_h;

    float* d_buf = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_buf, out_size));

    // rgb path
    cuframe::fused_nv12_to_tensor(nv12_ptr_, d_buf, w_, h_, pitch_,
                                   params, cuframe::BT601, cuframe::IMAGENET_NORM, false);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rgb(3 * plane);
    CUFRAME_CUDA_CHECK(cudaMemcpy(rgb.data(), d_buf, out_size, cudaMemcpyDeviceToHost));

    // bgr path
    cuframe::fused_nv12_to_tensor(nv12_ptr_, d_buf, w_, h_, pitch_,
                                   params, cuframe::BT601, cuframe::IMAGENET_NORM, true);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> bgr(3 * plane);
    CUFRAME_CUDA_CHECK(cudaMemcpy(bgr.data(), d_buf, out_size, cudaMemcpyDeviceToHost));

    // plane 0 of BGR == plane 2 of RGB, plane 2 of BGR == plane 0 of RGB
    for (int i = 0; i < plane; i += 37) {
        EXPECT_FLOAT_EQ(bgr[0 * plane + i], rgb[2 * plane + i])
            << "BGR plane 0 != RGB plane 2 at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[1 * plane + i], rgb[1 * plane + i])
            << "G plane differs at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[2 * plane + i], rgb[0 * plane + i])
            << "BGR plane 2 != RGB plane 0 at pixel " << i;
    }

    cudaFree(d_buf);
}

TEST_F(FusedPreprocessTest, StreamExecution) {
    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    const int dst_w = 640, dst_h = 640;
    auto params = cuframe::make_letterbox_params(w_, h_, dst_w, dst_h);
    size_t out_size = 3 * dst_w * dst_h * sizeof(float);

    float* fused_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&fused_d, out_size));

    cuframe::fused_nv12_to_tensor(nv12_ptr_, fused_d, w_, h_, pitch_,
                                   params, cuframe::BT601, cuframe::IMAGENET_NORM, stream);
    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> host(3 * dst_w * dst_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), fused_d, out_size, cudaMemcpyDeviceToHost));

    // ImageNet-normalized values should be roughly in [-3, 3]
    for (size_t i = 0; i < host.size(); i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    cudaFree(fused_d);
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));
}
