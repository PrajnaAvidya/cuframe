#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <filesystem>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

TEST(Demuxer, OpenAndReadInfo) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    auto& info = demuxer.video_info();

    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_EQ(info.codec_id, AV_CODEC_ID_H264);
    EXPECT_FALSE(info.extradata.empty());
    EXPECT_GT(info.time_base.den, 0);
}

TEST(Demuxer, ReadAllPackets) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    AVPacket* pkt = av_packet_alloc();

    int count = 0;
    while (demuxer.read_packet(pkt)) {
        EXPECT_NE(pkt->data, nullptr);
        EXPECT_GT(pkt->size, 0);
        count++;
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);

    // 3 seconds @ 30fps = ~90 frames, but encoding may produce slightly different counts
    EXPECT_GT(count, 60);
    EXPECT_LT(count, 150);
}

TEST(Demuxer, EofReturnsFalse) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    AVPacket* pkt = av_packet_alloc();

    // drain all packets
    while (demuxer.read_packet(pkt)) {
        av_packet_unref(pkt);
    }

    // subsequent reads should also return false
    EXPECT_FALSE(demuxer.read_packet(pkt));

    av_packet_free(&pkt);
}

TEST(Demuxer, NonexistentFileThrows) {
    EXPECT_THROW(cuframe::Demuxer("nonexistent.mp4"), std::runtime_error);
}
