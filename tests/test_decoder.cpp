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

// helper: demux + decode all frames from test video
static std::vector<cuframe::DecodedFrame> decode_all() {
    cuframe::Demuxer demuxer(TEST_VIDEO);
    cuframe::Decoder decoder(demuxer.video_info());

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> frames;

    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    decoder.flush(frames);
    return frames;
}

TEST(Decoder, DecodeAllFrames) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    auto frames = decode_all();

    // 3s @ 30fps = ~90 frames
    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);

    for (auto& f : frames) {
        EXPECT_NE(f.buffer.data(), nullptr);
        EXPECT_GT(f.buffer.size(), 0u);
        EXPECT_GT(f.width, 0);
        EXPECT_GT(f.height, 0);
        EXPECT_GT(f.pitch, 0u);
    }
}

TEST(Decoder, FrameDimensionsMatch) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    auto frames = decode_all();
    ASSERT_FALSE(frames.empty());

    // test video is 320x240
    EXPECT_EQ(frames[0].width, 320);
    EXPECT_EQ(frames[0].height, 240);
}

TEST(Decoder, FrameDataIsNonZero) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    auto frames = decode_all();
    ASSERT_FALSE(frames.empty());

    // copy first 64 bytes of luma back to host, verify not all zeros
    auto& f = frames[0];
    uint8_t host_buf[64] = {};
    cudaMemcpy(host_buf, f.buffer.data(), sizeof(host_buf), cudaMemcpyDeviceToHost);

    bool all_zero = true;
    for (auto b : host_buf) {
        if (b != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero) << "decoded frame luma data is all zeros";
}

TEST(Decoder, FlushProducesRemainingFrames) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    cuframe::Decoder decoder(demuxer.video_info());

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> frames;
    size_t decode_count = 0;

    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        decode_count = frames.size();
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    decoder.flush(frames);

    // flush should produce at least the same total (may add more due to B-frame reordering)
    EXPECT_GE(frames.size(), decode_count);

    // total should be reasonable
    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);
}
