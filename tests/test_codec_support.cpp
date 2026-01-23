#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/pipeline.h"
#include "cuframe/cuda_utils.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <cmath>

extern "C" {
#include <libavcodec/packet.h>
}

// test video paths (320x240, 30fps, 3s, ~90 frames each)
static const char* TEST_HEVC_MP4 = "tests/data/test_hevc.mp4";
static const char* TEST_HEVC_MKV = "tests/data/test_hevc.mkv";
static const char* TEST_VP9_WEBM = "tests/data/test_vp9.webm";
static const char* TEST_AV1_MP4  = "tests/data/test_av1.mp4";

static bool file_exists(const char* path) {
    return std::filesystem::exists(path);
}

// ============================================================================
// HEVC in mp4 container
// ============================================================================

TEST(CodecSupport, HEVC_MP4_Demuxer) {
    if (!file_exists(TEST_HEVC_MP4)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_HEVC_MP4);
    auto& info = demuxer.video_info();

    EXPECT_EQ(info.codec_id, AV_CODEC_ID_HEVC);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_FALSE(info.extradata.empty()) << "HEVC should have VPS/SPS/PPS extradata";
    EXPECT_GT(info.time_base.den, 0);

    AVPacket* pkt = av_packet_alloc();
    int count = 0;
    while (demuxer.read_packet(pkt)) {
        EXPECT_NE(pkt->data, nullptr);
        EXPECT_GT(pkt->size, 0);
        count++;
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    EXPECT_GT(count, 60);
    EXPECT_LT(count, 150);
}

TEST(CodecSupport, HEVC_MP4_Decoder) {
    if (!file_exists(TEST_HEVC_MP4)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_HEVC_MP4);
    cuframe::Decoder decoder(demuxer.video_info(), 150);

    std::vector<cuframe::DecodedFrame> frames;
    AVPacket* pkt = av_packet_alloc();
    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    decoder.flush(frames);

    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);

    for (auto& f : frames) {
        EXPECT_EQ(f.width, 320);
        EXPECT_EQ(f.height, 240);
        EXPECT_GT(f.pitch, 0u);
        EXPECT_NE(f.buffer->data(), nullptr);
    }

    // spot-check first frame data is non-zero
    uint8_t host_buf[64] = {};
    cudaMemcpy(host_buf, frames[0].buffer->data(), sizeof(host_buf),
               cudaMemcpyDeviceToHost);
    bool all_zero = true;
    for (auto b : host_buf) { if (b != 0) { all_zero = false; break; } }
    EXPECT_FALSE(all_zero) << "decoded HEVC frame data is all zeros";
}

TEST(CodecSupport, HEVC_MP4_PipelineFused) {
    if (!file_exists(TEST_HEVC_MP4)) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_HEVC_MP4)
        .resize(640, 640)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    while (auto batch = pipeline.next()) {
        auto& b = *batch;
        EXPECT_EQ(b->channels(), 3);
        EXPECT_EQ(b->height(), 640);
        EXPECT_EQ(b->width(), 640);
        EXPECT_GT(b->count(), 0);
        EXPECT_LE(b->count(), 8);

        size_t n = static_cast<size_t>(b->count()) * 3 * 640 * 640;
        std::vector<float> host(n);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                       n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i += 97) {
            EXPECT_GT(host[i], -5.0f);
            EXPECT_LT(host[i], 5.0f);
        }

        total_frames += b->count();
    }

    EXPECT_GT(total_frames, 60);
    EXPECT_LT(total_frames, 150);
}

TEST(CodecSupport, HEVC_MP4_PipelineColorConvert) {
    if (!file_exists(TEST_HEVC_MP4)) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_HEVC_MP4)
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    EXPECT_EQ(b->height(), 240);
    EXPECT_EQ(b->width(), 320);
    EXPECT_EQ(b->channels(), 3);
    EXPECT_EQ(b->count(), 4);

    size_t total = static_cast<size_t>(b->count()) * 3 * 240 * 320;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total; i += 97) {
        EXPECT_GE(host[i], -1.0f);
        EXPECT_LE(host[i], 256.0f);
    }
}

// ============================================================================
// HEVC in mkv container
// ============================================================================

TEST(CodecSupport, HEVC_MKV_Demuxer) {
    if (!file_exists(TEST_HEVC_MKV)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_HEVC_MKV);
    auto& info = demuxer.video_info();

    EXPECT_EQ(info.codec_id, AV_CODEC_ID_HEVC);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_FALSE(info.extradata.empty());
}

TEST(CodecSupport, HEVC_MKV_Decoder) {
    if (!file_exists(TEST_HEVC_MKV)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_HEVC_MKV);
    cuframe::Decoder decoder(demuxer.video_info(), 150);

    std::vector<cuframe::DecodedFrame> frames;
    AVPacket* pkt = av_packet_alloc();
    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    decoder.flush(frames);

    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);
    EXPECT_EQ(frames[0].width, 320);
    EXPECT_EQ(frames[0].height, 240);
}

TEST(CodecSupport, HEVC_MKV_PipelineFused) {
    if (!file_exists(TEST_HEVC_MKV)) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_HEVC_MKV)
        .resize(640, 640)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    while (auto batch = pipeline.next()) {
        EXPECT_EQ((*batch)->height(), 640);
        EXPECT_EQ((*batch)->width(), 640);
        total_frames += (*batch)->count();
    }

    EXPECT_GT(total_frames, 60);
    EXPECT_LT(total_frames, 150);
}

// ============================================================================
// VP9 in webm container
// ============================================================================

TEST(CodecSupport, VP9_WebM_Demuxer) {
    if (!file_exists(TEST_VP9_WEBM)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_VP9_WEBM);
    auto& info = demuxer.video_info();

    EXPECT_EQ(info.codec_id, AV_CODEC_ID_VP9);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_GT(info.time_base.den, 0);

    AVPacket* pkt = av_packet_alloc();
    int count = 0;
    while (demuxer.read_packet(pkt)) {
        count++;
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    EXPECT_GT(count, 60);
    EXPECT_LT(count, 150);
}

TEST(CodecSupport, VP9_WebM_Decoder) {
    if (!file_exists(TEST_VP9_WEBM)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_VP9_WEBM);
    cuframe::Decoder decoder(demuxer.video_info(), 150);

    std::vector<cuframe::DecodedFrame> frames;
    AVPacket* pkt = av_packet_alloc();
    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    decoder.flush(frames);

    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);
    EXPECT_EQ(frames[0].width, 320);
    EXPECT_EQ(frames[0].height, 240);

    // spot-check frame data
    uint8_t host_buf[64] = {};
    cudaMemcpy(host_buf, frames[0].buffer->data(), sizeof(host_buf),
               cudaMemcpyDeviceToHost);
    bool all_zero = true;
    for (auto b : host_buf) { if (b != 0) { all_zero = false; break; } }
    EXPECT_FALSE(all_zero) << "decoded VP9 frame data is all zeros";
}

TEST(CodecSupport, VP9_WebM_PipelineFused) {
    if (!file_exists(TEST_VP9_WEBM)) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VP9_WEBM)
        .resize(640, 640)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    while (auto batch = pipeline.next()) {
        auto& b = *batch;
        EXPECT_EQ(b->channels(), 3);
        EXPECT_EQ(b->height(), 640);
        EXPECT_EQ(b->width(), 640);

        size_t n = static_cast<size_t>(b->count()) * 3 * 640 * 640;
        std::vector<float> host(n);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                       n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i += 97) {
            EXPECT_GT(host[i], -5.0f);
            EXPECT_LT(host[i], 5.0f);
        }

        total_frames += b->count();
    }

    EXPECT_GT(total_frames, 60);
    EXPECT_LT(total_frames, 150);
}

TEST(CodecSupport, VP9_WebM_PipelineColorConvert) {
    if (!file_exists(TEST_VP9_WEBM)) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VP9_WEBM)
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    EXPECT_EQ(b->height(), 240);
    EXPECT_EQ(b->width(), 320);
    EXPECT_EQ(b->count(), 4);

    size_t total = static_cast<size_t>(b->count()) * 3 * 240 * 320;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total; i += 97) {
        EXPECT_GE(host[i], -1.0f);
        EXPECT_LE(host[i], 256.0f);
    }
}

// ============================================================================
// AV1 in mp4 container
// note: AV1 NVDEC requires Ada Lovelace (RTX 40 series) or newer.
// decoder/pipeline tests skip gracefully on unsupported hardware.
// ============================================================================

TEST(CodecSupport, AV1_MP4_Demuxer) {
    if (!file_exists(TEST_AV1_MP4)) GTEST_SKIP() << "test video not found";

    // demuxing works regardless of GPU capability
    cuframe::Demuxer demuxer(TEST_AV1_MP4);
    auto& info = demuxer.video_info();

    EXPECT_EQ(info.codec_id, AV_CODEC_ID_AV1);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_GT(info.time_base.den, 0);

    AVPacket* pkt = av_packet_alloc();
    int count = 0;
    while (demuxer.read_packet(pkt)) {
        count++;
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    EXPECT_GT(count, 60);
    EXPECT_LT(count, 150);
}

TEST(CodecSupport, AV1_MP4_Decoder) {
    if (!file_exists(TEST_AV1_MP4)) GTEST_SKIP() << "test video not found";

    cuframe::Demuxer demuxer(TEST_AV1_MP4);

    std::unique_ptr<cuframe::Decoder> decoder;
    try {
        decoder = std::make_unique<cuframe::Decoder>(demuxer.video_info(), 150);
    } catch (const std::runtime_error& e) {
        GTEST_SKIP() << "AV1 NVDEC not supported on this GPU: " << e.what();
    }

    std::vector<cuframe::DecodedFrame> frames;
    AVPacket* pkt = av_packet_alloc();

    // first decode call may fail if hardware doesn't support the codec
    // (on_sequence callback triggers cuvidCreateDecoder)
    try {
        while (demuxer.read_packet(pkt)) {
            decoder->decode(pkt, frames);
            av_packet_unref(pkt);
        }
        decoder->flush(frames);
    } catch (const std::runtime_error& e) {
        av_packet_free(&pkt);
        GTEST_SKIP() << "AV1 NVDEC not supported on this GPU: " << e.what();
    }
    av_packet_free(&pkt);

    EXPECT_GT(frames.size(), 60u);
    EXPECT_LT(frames.size(), 150u);
    EXPECT_EQ(frames[0].width, 320);
    EXPECT_EQ(frames[0].height, 240);
}

TEST(CodecSupport, AV1_MP4_PipelineFused) {
    if (!file_exists(TEST_AV1_MP4)) GTEST_SKIP() << "test video not found";

    std::unique_ptr<cuframe::Pipeline> pipeline;
    try {
        auto p = cuframe::Pipeline::builder()
            .input(TEST_AV1_MP4)
            .resize(640, 640)
            .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
            .batch(8)
            .build();
        pipeline = std::make_unique<cuframe::Pipeline>(std::move(p));
    } catch (const std::runtime_error& e) {
        GTEST_SKIP() << "AV1 NVDEC not supported on this GPU: " << e.what();
    }

    int total_frames = 0;
    try {
        while (auto batch = pipeline->next()) {
            EXPECT_EQ((*batch)->height(), 640);
            EXPECT_EQ((*batch)->width(), 640);
            total_frames += (*batch)->count();
        }
    } catch (const std::runtime_error& e) {
        GTEST_SKIP() << "AV1 NVDEC not supported on this GPU: " << e.what();
    }

    EXPECT_GT(total_frames, 60);
    EXPECT_LT(total_frames, 150);
}
