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

// ============================================================================
// ResizeParams factory tests (no GPU needed)
// ============================================================================

TEST(CenterCropParams, ResizeThenCrop) {
    auto p = cuframe::make_center_crop_params(320, 240, 256, 256, 224, 224);
    EXPECT_EQ(p.dst_w, 224);
    EXPECT_EQ(p.dst_h, 224);
    EXPECT_EQ(p.inner_w, 224);
    EXPECT_EQ(p.inner_h, 224);
    EXPECT_EQ(p.pad_left, 0);
    EXPECT_EQ(p.pad_top, 0);
    EXPECT_EQ(p.src_w, 320);
    EXPECT_EQ(p.src_h, 240);

    // scale = source/resized pixels per output pixel
    EXPECT_FLOAT_EQ(p.scale_x, 320.0f / 256.0f);
    EXPECT_FLOAT_EQ(p.scale_y, 240.0f / 256.0f);

    // crop origin in resized space: (16, 16), mapped to source: (16 * 320/256, 16 * 240/256)
    EXPECT_FLOAT_EQ(p.src_offset_x, 16.0f * 320.0f / 256.0f);
    EXPECT_FLOAT_EQ(p.src_offset_y, 16.0f * 240.0f / 256.0f);
}

TEST(CenterCropParams, CropOnly) {
    auto p = cuframe::make_center_crop_params(320, 240, 0, 0, 200, 200);
    EXPECT_EQ(p.dst_w, 200);
    EXPECT_EQ(p.dst_h, 200);
    EXPECT_FLOAT_EQ(p.scale_x, 1.0f);
    EXPECT_FLOAT_EQ(p.scale_y, 1.0f);
    EXPECT_FLOAT_EQ(p.src_offset_x, 60.0f);
    EXPECT_FLOAT_EQ(p.src_offset_y, 20.0f);
}

TEST(CenterCropParams, SrcOffsetDefaultZero) {
    auto stretch = cuframe::make_resize_params(320, 240, 640, 640);
    EXPECT_FLOAT_EQ(stretch.src_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(stretch.src_offset_y, 0.0f);

    auto letterbox = cuframe::make_letterbox_params(320, 240, 640, 640);
    EXPECT_FLOAT_EQ(letterbox.src_offset_x, 0.0f);
    EXPECT_FLOAT_EQ(letterbox.src_offset_y, 0.0f);
}

// ============================================================================
// GPU tests — decode a frame, test crop kernels
// ============================================================================

class CenterCropTest : public ::testing::Test {
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

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;

    const uint8_t* nv12_ptr_ = nullptr;
    int w_ = 0, h_ = 0;
    unsigned int pitch_ = 0;
};

TEST_F(CenterCropTest, FusedCropOnlyPixelValues) {
    // crop 200x200 from center of 320x240 source
    int crop_w = 200, crop_h = 200;
    auto params = cuframe::make_center_crop_params(w_, h_, 0, 0, crop_w, crop_h);
    size_t out_size = 3 * crop_w * crop_h * sizeof(float);

    float* d_out = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_out, out_size));

    cuframe::fused_nv12_to_tensor(nv12_ptr_, d_out, w_, h_, pitch_,
                                   params, cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(3 * crop_w * crop_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), d_out, out_size, cudaMemcpyDeviceToHost));

    // values should be in ImageNet normalized range
    for (size_t i = 0; i < host.size(); i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    // non-trivial content
    float first = host[0];
    bool all_same = true;
    for (size_t i = 1; i < host.size(); i += 31) {
        if (std::abs(host[i] - first) > 1e-5f) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same) << "crop output is all identical values";

    cudaFree(d_out);
}

TEST_F(CenterCropTest, CropMatchesManualExtract) {
    // run full-frame color convert, then manually extract center crop on CPU.
    // compare against fused crop-only kernel.
    int crop_w = 200, crop_h = 160;
    int offset_x = (w_ - crop_w) / 2;
    int offset_y = (h_ - crop_h) / 2;

    // full-frame color convert
    size_t full_size = 3 * w_ * h_ * sizeof(float);
    float* d_full = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_full, full_size));
    cuframe::nv12_to_rgb_planar(nv12_ptr_, d_full, w_, h_, pitch_, cuframe::BT601);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> full_host(3 * w_ * h_);
    CUFRAME_CUDA_CHECK(cudaMemcpy(full_host.data(), d_full, full_size, cudaMemcpyDeviceToHost));
    cudaFree(d_full);

    // manually extract center crop on CPU
    std::vector<float> manual_crop(3 * crop_w * crop_h);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < crop_h; ++y) {
            for (int x = 0; x < crop_w; ++x) {
                int src_idx = c * w_ * h_ + (y + offset_y) * w_ + (x + offset_x);
                int dst_idx = c * crop_w * crop_h + y * crop_w + x;
                manual_crop[dst_idx] = full_host[src_idx];
            }
        }
    }

    // fused crop (no normalize — use identity norm to get raw [0,255] values)
    // actually fused requires norm, so use scale=1/255, bias=0 to get [0,1] range,
    // then compare against manual_crop scaled the same way.
    // simpler: use the separate resize path with crop params to get unnormalized output
    auto crop_params = cuframe::make_center_crop_params(w_, h_, 0, 0, crop_w, crop_h);
    size_t crop_size = 3 * crop_w * crop_h * sizeof(float);

    // use resize_bilinear with crop params on the full RGB
    float* d_full2 = nullptr;
    float* d_crop = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_full2, full_size));
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_crop, crop_size));

    cuframe::nv12_to_rgb_planar(nv12_ptr_, d_full2, w_, h_, pitch_, cuframe::BT601);
    cuframe::resize_bilinear(d_full2, d_crop, crop_params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernel_crop(3 * crop_w * crop_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(kernel_crop.data(), d_crop, crop_size, cudaMemcpyDeviceToHost));

    cudaFree(d_full2);
    cudaFree(d_crop);

    // compare — scale=1.0 crop should match pixel-for-pixel with manual crop
    // (bilinear at scale 1.0 with integer offsets should reproduce exact values)
    int mismatch = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < manual_crop.size(); ++i) {
        float diff = std::abs(manual_crop[i] - kernel_crop[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.6f) ++mismatch;  // bilinear at 1:1 scale can have slight interpolation diffs at edges
    }

    EXPECT_EQ(mismatch, 0)
        << mismatch << " mismatches out of " << manual_crop.size()
        << ", max diff = " << max_diff;
}

TEST_F(CenterCropTest, ResizeThenCropFused) {
    // resize 256x256 + center crop 224x224
    int crop_w = 224, crop_h = 224;
    auto params = cuframe::make_center_crop_params(w_, h_, 256, 256, crop_w, crop_h);
    size_t out_size = 3 * crop_w * crop_h * sizeof(float);

    float* d_out = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&d_out, out_size));

    cuframe::fused_nv12_to_tensor(nv12_ptr_, d_out, w_, h_, pitch_,
                                   params, cuframe::BT601, cuframe::IMAGENET_NORM);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(3 * crop_w * crop_h);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), d_out, out_size, cudaMemcpyDeviceToHost));

    // valid normalized range
    for (size_t i = 0; i < host.size(); i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    // non-trivial content
    float first = host[0];
    bool all_same = true;
    for (size_t i = 1; i < host.size(); i += 31) {
        if (std::abs(host[i] - first) > 1e-5f) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same) << "resize+crop output is all identical values";

    cudaFree(d_out);
}
