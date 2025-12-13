#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <cstring>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture keeps decoder alive so pooled buffers remain valid.
// destruction order: frames_ first (returns buffers), then decoder_ (frees pool).
class DecoderTest : public ::testing::Test {
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

TEST_F(DecoderTest, DecodeAllFrames) {
    decode_all();

    // 3s @ 30fps = ~90 frames
    EXPECT_GT(frames_.size(), 60u);
    EXPECT_LT(frames_.size(), 150u);

    for (auto& f : frames_) {
        EXPECT_NE(f.buffer->data(), nullptr);
        EXPECT_GT(f.buffer->size(), 0u);
        EXPECT_GT(f.width, 0);
        EXPECT_GT(f.height, 0);
        EXPECT_GT(f.pitch, 0u);
    }
}

TEST_F(DecoderTest, FrameDimensionsMatch) {
    decode_all();
    ASSERT_FALSE(frames_.empty());

    // test video is 320x240
    EXPECT_EQ(frames_[0].width, 320);
    EXPECT_EQ(frames_[0].height, 240);
}

TEST_F(DecoderTest, FrameDataIsNonZero) {
    decode_all();
    ASSERT_FALSE(frames_.empty());

    // copy first 64 bytes of luma back to host, verify not all zeros
    auto& f = frames_[0];
    uint8_t host_buf[64] = {};
    cudaMemcpy(host_buf, f.buffer->data(), sizeof(host_buf), cudaMemcpyDeviceToHost);

    bool all_zero = true;
    for (auto b : host_buf) {
        if (b != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero) << "decoded frame luma data is all zeros";
}

TEST_F(DecoderTest, FlushProducesRemainingFrames) {
    AVPacket* pkt = av_packet_alloc();
    size_t decode_count = 0;

    while (demuxer_->read_packet(pkt)) {
        decoder_->decode(pkt, frames_);
        decode_count = frames_.size();
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    decoder_->flush(frames_);

    // flush should produce at least the same total (may add more due to B-frame reordering)
    EXPECT_GE(frames_.size(), decode_count);

    // total should be reasonable
    EXPECT_GT(frames_.size(), 60u);
    EXPECT_LT(frames_.size(), 150u);
}
