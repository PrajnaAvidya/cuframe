#include "cuframe/pipeline.h"
#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cmath>
#include <chrono>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";
static const int DST_W = 640;
static const int DST_H = 640;

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// ============================================================================
// correctness: total frame count matches expected (prefetch doesn't drop/dup)
// ============================================================================

TEST(PipelineMultistreamTest, FullIterationCorrectness) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    int batch_count = 0;

    while (auto batch = pipeline.next()) {
        auto& b = *batch;
        EXPECT_EQ(b->channels(), 3);
        EXPECT_EQ(b->height(), DST_H);
        EXPECT_EQ(b->width(), DST_W);
        EXPECT_GT(b->count(), 0);
        EXPECT_LE(b->count(), 8);
        total_frames += b->count();
        batch_count++;
    }

    EXPECT_EQ(total_frames, 90);
    EXPECT_EQ(batch_count, 12);

    // exhausted pipeline returns nullopt
    EXPECT_FALSE(pipeline.next().has_value());
}

// ============================================================================
// correctness: output matches direct kernel calls
// ============================================================================

TEST(PipelineMultistreamTest, OutputMatchesDirectKernels) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    const int BATCH_SIZE = 8;

    // pipeline path (with prefetch/multi-stream)
    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(BATCH_SIZE)
        .build();

    auto pipeline_batch = pipeline.next();
    ASSERT_TRUE(pipeline_batch.has_value());
    EXPECT_EQ((*pipeline_batch)->count(), BATCH_SIZE);

    // direct kernel path
    cuframe::Demuxer demuxer(TEST_VIDEO);
    auto& info = demuxer.video_info();
    cuframe::Decoder decoder(info, 20);
    auto resize_params = cuframe::make_letterbox_params(
        info.width, info.height, DST_W, DST_H);

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> frames;
    while (demuxer.read_packet(pkt) && static_cast<int>(frames.size()) < BATCH_SIZE) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    ASSERT_GE(static_cast<int>(frames.size()), BATCH_SIZE);

    size_t out_bytes = 3ULL * DST_W * DST_H * sizeof(float);
    std::vector<float*> out_bufs(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUFRAME_CUDA_CHECK(cudaMalloc(&out_bufs[i], out_bytes));
        auto& f = frames[i];
        auto* nv12 = static_cast<const uint8_t*>(f.buffer->data());
        cuframe::fused_nv12_to_tensor(nv12, out_bufs[i],
            f.width, f.height, f.pitch,
            resize_params, cuframe::BT601, cuframe::IMAGENET_NORM);
    }

    cuframe::GpuFrameBatch direct_batch(BATCH_SIZE, 3, DST_H, DST_W);
    std::vector<const float*> ptrs(out_bufs.begin(), out_bufs.end());
    cuframe::batch_frames(direct_batch, ptrs.data(), BATCH_SIZE);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // compare
    size_t total = static_cast<size_t>(BATCH_SIZE) * 3 * DST_H * DST_W;
    std::vector<float> pipeline_host(total), direct_host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(pipeline_host.data(), (*pipeline_batch)->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));
    CUFRAME_CUDA_CHECK(cudaMemcpy(direct_host.data(), direct_batch.data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatch = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        float diff = std::abs(pipeline_host[i] - direct_host[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) ++mismatch;
    }

    EXPECT_EQ(mismatch, 0)
        << mismatch << " mismatches out of " << total
        << ", max diff = " << max_diff;

    for (auto* p : out_bufs) cudaFree(p);
}

// ============================================================================
// partial last batch with prefetch + flush
// ============================================================================

TEST(PipelineMultistreamTest, PartialLastBatch) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    int last_count = 0;

    while (auto batch = pipeline.next()) {
        last_count = (*batch)->count();
        total_frames += last_count;
    }

    EXPECT_EQ(total_frames, 90);
    // 90 frames / batch 8 = 11 full + 2 remainder
    EXPECT_EQ(last_count, 2);
}

// ============================================================================
// prefetch observation: second next() call benefits from pre-decoded frames
// ============================================================================

TEST(PipelineMultistreamTest, PrefetchBenefit) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    using Clock = std::chrono::steady_clock;

    // first call: cold start (decode + preprocess + prefetch)
    auto t0 = Clock::now();
    auto batch1 = pipeline.next();
    auto t1 = Clock::now();
    ASSERT_TRUE(batch1.has_value());

    // second call: should find pre-decoded frames (less decode wait)
    auto t2 = Clock::now();
    auto batch2 = pipeline.next();
    auto t3 = Clock::now();
    ASSERT_TRUE(batch2.has_value());

    double first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double second_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // just print — timing assertions are too flaky for CI
    printf("  first next():  %.2f ms\n", first_ms);
    printf("  second next(): %.2f ms\n", second_ms);
    printf("  speedup:       %.2fx\n", first_ms / second_ms);
}
