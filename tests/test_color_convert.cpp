#include "cuframe/kernels/color_convert.h"
#include "cuframe/cuda_utils.h"
#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture: decode first frame, allocate rgb output buffer.
// destruction order: frames_ → decoder_ (pool must outlive frames).
class ColorConvertTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found";
        demuxer_ = std::make_unique<cuframe::Demuxer>(TEST_VIDEO);
        decoder_ = std::make_unique<cuframe::Decoder>(demuxer_->video_info(), 10);
        decode_first_frame();

        w_ = frames_[0].width;
        h_ = frames_[0].height;
        pitch_ = frames_[0].pitch;
        pixels_ = w_ * h_;
        rgb_size_ = 3 * pixels_ * sizeof(float);

        CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_ptr_, rgb_size_));
    }

    void TearDown() override {
        if (rgb_ptr_) cudaFree(rgb_ptr_);
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

    const cuframe::DecodedFrame& frame() const { return frames_[0]; }

    // copy rgb output to host
    std::vector<float> rgb_to_host() {
        std::vector<float> host(3 * pixels_);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), rgb_ptr_, rgb_size_, cudaMemcpyDeviceToHost));
        return host;
    }

    // copy nv12 input to host
    std::vector<uint8_t> nv12_to_host() {
        auto& f = frame();
        std::vector<uint8_t> host(f.buffer->size());
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), f.buffer->data(), f.buffer->size(), cudaMemcpyDeviceToHost));
        return host;
    }

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;

    float* rgb_ptr_ = nullptr;
    size_t rgb_size_ = 0;
    int w_ = 0, h_ = 0, pixels_ = 0;
    unsigned int pitch_ = 0;
};

TEST_F(ColorConvertTest, OutputDimensions) {
    auto& f = frame();
    cuframe::nv12_to_rgb_planar(
        static_cast<const uint8_t*>(f.buffer->data()), rgb_ptr_,
        w_, h_, pitch_, cuframe::BT601
    );
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto rgb = rgb_to_host();
    EXPECT_EQ(rgb.size(), static_cast<size_t>(3 * pixels_));
}

TEST_F(ColorConvertTest, ValuesInRange) {
    auto& f = frame();
    cuframe::nv12_to_rgb_planar(
        static_cast<const uint8_t*>(f.buffer->data()), rgb_ptr_,
        w_, h_, pitch_, cuframe::BT601
    );
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto rgb = rgb_to_host();
    for (size_t i = 0; i < rgb.size(); ++i) {
        EXPECT_GE(rgb[i], 0.0f) << "negative value at index " << i;
        EXPECT_LE(rgb[i], 255.0f) << "value > 255 at index " << i;
    }
}

TEST_F(ColorConvertTest, NonTrivialContent) {
    auto& f = frame();
    cuframe::nv12_to_rgb_planar(
        static_cast<const uint8_t*>(f.buffer->data()), rgb_ptr_,
        w_, h_, pitch_, cuframe::BT601
    );
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto rgb = rgb_to_host();

    // check that we have real content, not all zeros or all same value
    float first = rgb[0];
    bool all_same = true;
    bool all_zero = true;
    for (size_t i = 0; i < rgb.size(); i += 37) {  // sample every 37th pixel
        if (rgb[i] != first) all_same = false;
        if (rgb[i] != 0.0f) all_zero = false;
    }
    EXPECT_FALSE(all_zero) << "output is all zeros";
    EXPECT_FALSE(all_same) << "output is all identical values";
}

TEST_F(ColorConvertTest, SpotCheckPixels) {
    auto& f = frame();
    cuframe::nv12_to_rgb_planar(
        static_cast<const uint8_t*>(f.buffer->data()), rgb_ptr_,
        w_, h_, pitch_, cuframe::BT601
    );
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto nv12 = nv12_to_host();
    auto rgb = rgb_to_host();

    // spot check a few pixels against manual BT.601 computation
    int test_coords[][2] = {{10, 10}, {100, 50}, {w_ / 2, h_ / 2}, {0, 0}};

    for (auto& coord : test_coords) {
        int px = coord[0], py = coord[1];
        if (px >= w_ || py >= h_) continue;

        float Y = static_cast<float>(nv12[py * pitch_ + px]);
        int chroma_off = pitch_ * h_;
        float U = static_cast<float>(nv12[chroma_off + (py / 2) * pitch_ + (px / 2) * 2]) - 128.0f;
        float V = static_cast<float>(nv12[chroma_off + (py / 2) * pitch_ + (px / 2) * 2 + 1]) - 128.0f;

        auto clampf = [](float v) { return std::min(std::max(v, 0.0f), 255.0f); };
        float expected_r = clampf(Y * 1.0f + U * 0.0f     + V * 1.402f);
        float expected_g = clampf(Y * 1.0f + U * -0.344136f + V * -0.714136f);
        float expected_b = clampf(Y * 1.0f + U * 1.772f    + V * 0.0f);

        int pixel = py * w_ + px;
        float actual_r = rgb[0 * pixels_ + pixel];
        float actual_g = rgb[1 * pixels_ + pixel];
        float actual_b = rgb[2 * pixels_ + pixel];

        EXPECT_NEAR(actual_r, expected_r, 1.0f) << "R mismatch at (" << px << "," << py << ")";
        EXPECT_NEAR(actual_g, expected_g, 1.0f) << "G mismatch at (" << px << "," << py << ")";
        EXPECT_NEAR(actual_b, expected_b, 1.0f) << "B mismatch at (" << px << "," << py << ")";
    }
}

TEST_F(ColorConvertTest, BT709DifferentFromBT601) {
    auto& f = frame();
    auto* nv12_ptr = static_cast<const uint8_t*>(f.buffer->data());

    // run BT601
    cuframe::nv12_to_rgb_planar(nv12_ptr, rgb_ptr_, w_, h_, pitch_, cuframe::BT601);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
    auto rgb601 = rgb_to_host();

    // run BT709 into same buffer
    cuframe::nv12_to_rgb_planar(nv12_ptr, rgb_ptr_, w_, h_, pitch_, cuframe::BT709);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
    auto rgb709 = rgb_to_host();

    // matrices differ, so outputs should differ for non-grayscale content
    int diff_count = 0;
    for (size_t i = 0; i < rgb601.size(); i += 13) {
        if (std::abs(rgb601[i] - rgb709[i]) > 0.01f) ++diff_count;
    }
    EXPECT_GT(diff_count, 0) << "BT601 and BT709 produced identical output";
}

TEST_F(ColorConvertTest, StreamExecution) {
    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    auto& f = frame();
    cuframe::nv12_to_rgb_planar(
        static_cast<const uint8_t*>(f.buffer->data()), rgb_ptr_,
        w_, h_, pitch_, cuframe::BT601, stream
    );
    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto rgb = rgb_to_host();

    // same checks as ValuesInRange
    for (size_t i = 0; i < rgb.size(); i += 97) {
        EXPECT_GE(rgb[i], 0.0f);
        EXPECT_LE(rgb[i], 255.0f);
    }

    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));
}
