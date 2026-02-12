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

TEST(Demuxer, SeekToStart) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    demuxer.seek(0.0);

    AVPacket* pkt = av_packet_alloc();
    EXPECT_TRUE(demuxer.read_packet(pkt));
    // first packet should be at or near the start
    auto& info = demuxer.video_info();
    double pts_sec = static_cast<double>(pkt->pts) * info.time_base.num / info.time_base.den;
    EXPECT_LT(pts_sec, 0.1);
    av_packet_unref(pkt);
    av_packet_free(&pkt);
}

TEST(Demuxer, SeekToMiddle) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    demuxer.seek(1.5);

    AVPacket* pkt = av_packet_alloc();
    EXPECT_TRUE(demuxer.read_packet(pkt));
    // first packet should be at a keyframe at or before 1.5s
    auto& info = demuxer.video_info();
    double pts_sec = static_cast<double>(pkt->pts) * info.time_base.num / info.time_base.den;
    EXPECT_LE(pts_sec, 1.5 + 0.1);  // at or slightly after target
    av_packet_unref(pkt);
    av_packet_free(&pkt);
}

TEST(Demuxer, SeekPastEnd) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    demuxer.seek(10.0);  // video is 3 seconds

    AVPacket* pkt = av_packet_alloc();
    bool got = demuxer.read_packet(pkt);
    // seeking past end: av_seek_frame may land on the last keyframe,
    // so we might still get packets — that's valid behavior
    if (got) av_packet_unref(pkt);
    av_packet_free(&pkt);
}

TEST(Demuxer, SeekThenSeekAgain) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found: " << TEST_VIDEO;

    cuframe::Demuxer demuxer(TEST_VIDEO);
    AVPacket* pkt = av_packet_alloc();

    demuxer.seek(2.0);
    EXPECT_TRUE(demuxer.read_packet(pkt));
    av_packet_unref(pkt);

    demuxer.seek(0.5);
    EXPECT_TRUE(demuxer.read_packet(pkt));
    av_packet_unref(pkt);

    av_packet_free(&pkt);
}
