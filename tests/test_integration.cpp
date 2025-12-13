#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture keeps decoder alive so pooled buffers remain valid.
// destruction order: frames_ first (returns buffers), then decoder_ (frees pool).
class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;
        demuxer_ = std::make_unique<cuframe::Demuxer>(TEST_VIDEO);
        decoder_ = std::make_unique<cuframe::Decoder>(demuxer_->video_info(), 150);
    }

    void decode_all() {
        AVPacket* pkt = av_packet_alloc();
        while (demuxer_->read_packet(pkt)) {
            decoder_->decode(pkt, frames_);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
        decoder_->flush(frames_);
    }

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;
};

TEST_F(IntegrationTest, DecodeFullVideo) {
    decode_all();

    // test video is 320x240, 3s @ 30fps = 90 frames
    EXPECT_EQ(frames_.size(), 90u);

    for (size_t i = 0; i < frames_.size(); ++i) {
        auto& f = frames_[i];
        EXPECT_NE(f.buffer->data(), nullptr) << "frame " << i;
        EXPECT_GT(f.buffer->size(), 0u) << "frame " << i;
        EXPECT_EQ(f.width, 320) << "frame " << i;
        EXPECT_EQ(f.height, 240) << "frame " << i;
        EXPECT_GT(f.pitch, 0u) << "frame " << i;
    }
}

TEST_F(IntegrationTest, TimestampsAreOrdered) {
    decode_all();
    ASSERT_GT(frames_.size(), 1u);

    // display-order timestamps should be non-decreasing
    for (size_t i = 1; i < frames_.size(); ++i) {
        EXPECT_GE(frames_[i].timestamp, frames_[i - 1].timestamp)
            << "timestamp regression at frame " << i;
    }
}

TEST_F(IntegrationTest, DumpFirstFrameNV12) {
    decode_all();
    ASSERT_FALSE(frames_.empty());

    auto& f = frames_[0];
    int w = f.width;
    int h = f.height;
    unsigned int pitch = f.pitch;

    // copy full pitched frame from gpu to host
    std::vector<uint8_t> gpu_data(f.buffer->size());
    cudaMemcpy(gpu_data.data(), f.buffer->data(), f.buffer->size(), cudaMemcpyDeviceToHost);

    // write packed NV12 (strip pitch padding) so ffplay can read it
    std::string dump_path = "tests/data/frame_dump.nv12";
    std::ofstream out(dump_path, std::ios::binary);
    ASSERT_TRUE(out.is_open()) << "failed to open " << dump_path;

    // luma plane: height rows, width bytes each
    for (int row = 0; row < h; ++row) {
        out.write(reinterpret_cast<const char*>(gpu_data.data() + row * pitch), w);
    }

    // chroma plane: height/2 rows, width bytes each (interleaved UV)
    int chroma_h = (h + 1) / 2;
    size_t chroma_offset = static_cast<size_t>(pitch) * h;
    for (int row = 0; row < chroma_h; ++row) {
        out.write(reinterpret_cast<const char*>(gpu_data.data() + chroma_offset + row * pitch), w);
    }
    out.close();

    // verify file size
    auto file_size = std::filesystem::file_size(dump_path);
    size_t expected = static_cast<size_t>(w) * h * 3 / 2;
    EXPECT_EQ(file_size, expected);

    printf("frame dumped to %s\n", dump_path.c_str());
    printf("view with: ffplay -f rawvideo -pix_fmt nv12 -video_size %dx%d %s\n",
           w, h, dump_path.c_str());
}
