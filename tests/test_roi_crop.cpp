#include "cuframe/kernels/roi_crop.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/pipeline.h"
#include "cuframe/batch_pool.h"
#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include "cuframe/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cmath>
#include <cstdint>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture: decode first frame, provide NV12 data
class RoiCropTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found";
        demuxer_ = std::make_unique<cuframe::Demuxer>(TEST_VIDEO);
        decoder_ = std::make_unique<cuframe::Decoder>(demuxer_->video_info(), 10);
        decode_first_frame();

        w_ = frames_[0].width;
        h_ = frames_[0].height;
        pitch_ = frames_[0].pitch;
        nv12_ = static_cast<const uint8_t*>(frames_[0].buffer->data());
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

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;

    const uint8_t* nv12_ = nullptr;
    int w_ = 0, h_ = 0;
    unsigned int pitch_ = 0;
};

// ============================================================================
// single ROI covering full frame matches fused kernel
// ============================================================================

TEST_F(RoiCropTest, SingleRoiMatchesFused) {
    const int dst_w = 128, dst_h = 128;
    size_t out_bytes = 3ULL * dst_w * dst_h * sizeof(float);

    // fused kernel path
    auto params = cuframe::make_resize_params(w_, h_, dst_w, dst_h);
    float* fused_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&fused_d, out_bytes));
    cuframe::fused_nv12_to_tensor(nv12_, fused_d, w_, h_, pitch_,
        params, cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // ROI crop path — full frame as single ROI
    cuframe::Rect roi{0, 0, w_, h_};
    float* crop_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&crop_d, out_bytes));
    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        &roi, 1, crop_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // compare
    size_t n = 3 * dst_w * dst_h;
    std::vector<float> fused_h(n), crop_h(n);
    CUFRAME_CUDA_CHECK(cudaMemcpy(fused_h.data(), fused_d, out_bytes, cudaMemcpyDeviceToHost));
    CUFRAME_CUDA_CHECK(cudaMemcpy(crop_h.data(), crop_d, out_bytes, cudaMemcpyDeviceToHost));

    int mismatch = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(fused_h[i] - crop_h[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) ++mismatch;
    }
    EXPECT_EQ(mismatch, 0)
        << mismatch << " mismatches out of " << n << ", max diff = " << max_diff;

    cudaFree(fused_d);
    cudaFree(crop_d);
}

// ============================================================================
// multiple ROIs produce non-trivial, distinct content
// ============================================================================

TEST_F(RoiCropTest, MultipleRois) {
    const int dst_w = 64, dst_h = 64;
    const int num_rois = 3;

    // three non-overlapping ROIs in different parts of the frame
    cuframe::Rect rois[3] = {
        {0, 0, w_ / 2, h_ / 2},                     // top-left quarter
        {w_ / 2, 0, w_ / 2, h_ / 2},                // top-right quarter
        {w_ / 4, h_ / 2, w_ / 2, h_ / 2},           // bottom-center
    };

    size_t crop_bytes = 3ULL * dst_w * dst_h * sizeof(float);
    size_t total_bytes = num_rois * crop_bytes;
    float* output_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&output_d, total_bytes));

    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        rois, num_rois, output_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(num_rois * 3 * dst_w * dst_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), output_d, total_bytes, cudaMemcpyDeviceToHost));

    size_t crop_floats = 3 * dst_w * dst_h;
    for (int r = 0; r < num_rois; ++r) {
        float* crop = host.data() + r * crop_floats;

        // values in ImageNet normalized range
        for (size_t i = 0; i < crop_floats; i += 37) {
            EXPECT_GT(crop[i], -5.0f) << "roi " << r << " pixel " << i;
            EXPECT_LT(crop[i], 5.0f) << "roi " << r << " pixel " << i;
        }

        // not all same value
        bool all_same = true;
        float first = crop[0];
        for (size_t i = 1; i < crop_floats; i += 31) {
            if (std::abs(crop[i] - first) > 1e-5f) { all_same = false; break; }
        }
        EXPECT_FALSE(all_same) << "roi " << r << " has all identical values";
    }

    // crops from different regions should differ
    float* crop0 = host.data();
    float* crop1 = host.data() + crop_floats;
    int differ = 0;
    for (size_t i = 0; i < crop_floats; i += 7)
        if (std::abs(crop0[i] - crop1[i]) > 1e-4f) ++differ;
    EXPECT_GT(differ, 0) << "ROI 0 and ROI 1 should produce different output";

    cudaFree(output_d);
}

// ============================================================================
// ROI at frame boundary — clamps correctly
// ============================================================================

TEST_F(RoiCropTest, RoiAtBoundary) {
    const int dst_w = 32, dst_h = 32;
    size_t out_bytes = 3ULL * dst_w * dst_h * sizeof(float);

    // ROI touching bottom-right corner
    cuframe::Rect roi{w_ - 40, h_ - 30, 40, 30};
    float* output_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&output_d, out_bytes));

    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        &roi, 1, output_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(3 * dst_w * dst_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), output_d, out_bytes, cudaMemcpyDeviceToHost));

    // should produce valid values (no NaN, no garbage)
    for (size_t i = 0; i < host.size(); i += 13) {
        EXPECT_FALSE(std::isnan(host[i])) << "NaN at " << i;
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    cudaFree(output_d);
}

// ============================================================================
// small ROI upscaled — bilinear produces smooth output
// ============================================================================

TEST_F(RoiCropTest, SmallRoiUpscaled) {
    const int dst_w = 64, dst_h = 64;
    size_t out_bytes = 3ULL * dst_w * dst_h * sizeof(float);

    // tiny 16x16 ROI from center, upscaled to 64x64
    cuframe::Rect roi{w_ / 2 - 8, h_ / 2 - 8, 16, 16};
    float* output_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&output_d, out_bytes));

    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        &roi, 1, output_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(3 * dst_w * dst_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), output_d, out_bytes, cudaMemcpyDeviceToHost));

    // all values finite and in range
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FALSE(std::isnan(host[i])) << "NaN at " << i;
        EXPECT_GT(host[i], -5.0f);
        EXPECT_LT(host[i], 5.0f);
    }

    cudaFree(output_d);
}

// ============================================================================
// zero ROIs — no crash
// ============================================================================

TEST_F(RoiCropTest, ZeroRois) {
    float* output_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&output_d, 1024));

    // should return immediately, no crash
    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        nullptr, 0, output_d, 64, 64,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(output_d);
}

// ============================================================================
// BGR output — planes swap correctly
// ============================================================================

TEST_F(RoiCropTest, BGROutput) {
    const int dst_w = 64, dst_h = 64;
    size_t out_bytes = 3ULL * dst_w * dst_h * sizeof(float);
    int plane = dst_w * dst_h;

    cuframe::Rect roi{0, 0, w_ / 2, h_ / 2};

    float* rgb_d = nullptr;
    float* bgr_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_d, out_bytes));
    CUFRAME_CUDA_CHECK(cudaMalloc(&bgr_d, out_bytes));

    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        &roi, 1, rgb_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM, false);
    cuframe::roi_crop_batch(nv12_, w_, h_, pitch_,
        &roi, 1, bgr_d, dst_w, dst_h,
        cuframe::BT601, cuframe::IMAGENET_NORM, true);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> rgb(3 * plane), bgr(3 * plane);
    CUFRAME_CUDA_CHECK(cudaMemcpy(rgb.data(), rgb_d, out_bytes, cudaMemcpyDeviceToHost));
    CUFRAME_CUDA_CHECK(cudaMemcpy(bgr.data(), bgr_d, out_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < plane; i += 37) {
        EXPECT_FLOAT_EQ(bgr[0 * plane + i], rgb[2 * plane + i])
            << "BGR plane 0 != RGB plane 2 at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[1 * plane + i], rgb[1 * plane + i])
            << "G plane differs at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[2 * plane + i], rgb[0 * plane + i])
            << "BGR plane 2 != RGB plane 0 at pixel " << i;
    }

    cudaFree(rgb_d);
    cudaFree(bgr_d);
}

// ============================================================================
// two-stage integration: pipeline + retain + roi_crop_batch
// ============================================================================

TEST(RoiCropIntegration, TwoStagePipeline) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    // stage 1: detector pipeline
    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .retain_decoded(true)
        .batch(1)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(pipeline.retained_count(), 1);

    // get retained NV12 frame
    auto& frame = pipeline.retained_frame(0);
    ASSERT_NE(frame.data, nullptr);

    // simulate detector output: 4 bounding boxes
    cuframe::Rect rois[] = {
        {10, 10, 100, 80},
        {150, 50, 60, 120},
        {0, 0, frame.width / 2, frame.height / 2},
        {frame.width / 2, frame.height / 2,
         frame.width / 2, frame.height / 2},
    };
    int num_rois = 4;
    int crop_w = 224, crop_h = 224;

    // stage 2: crop detections using BatchPool for output
    cuframe::BatchPool crop_pool(2, 16, 3, crop_h, crop_w);
    auto crops = crop_pool.acquire();

    cuframe::roi_crop_batch(
        frame.data, frame.width, frame.height, frame.pitch,
        rois, num_rois,
        crops->data(), crop_w, crop_h,
        cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
    crops->set_count(num_rois);

    EXPECT_EQ(crops->count(), 4);
    EXPECT_EQ(crops->height(), crop_h);
    EXPECT_EQ(crops->width(), crop_w);

    // verify crop values
    size_t total = static_cast<size_t>(num_rois) * 3 * crop_h * crop_w;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), crops->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < total; i += 97) {
        EXPECT_FALSE(std::isnan(host[i])) << "NaN at " << i;
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }
}
