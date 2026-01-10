#include "cuframe/pipeline.h"
#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
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
// full iteration: verify total frame count, partial last batch
// ============================================================================

TEST(PipelineTest, FullIteration) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    int batch_count = 0;
    int last_count = 0;

    while (auto batch = pipeline.next()) {
        auto& b = *batch;
        EXPECT_EQ(b->channels(), 3);
        EXPECT_EQ(b->height(), DST_H);
        EXPECT_EQ(b->width(), DST_W);
        EXPECT_EQ(b->batch_size(), 8);
        EXPECT_GT(b->count(), 0);
        EXPECT_LE(b->count(), 8);

        total_frames += b->count();
        last_count = b->count();
        batch_count++;
    }

    // test video: 320x240, 90 frames → 11 full batches + 1 partial (2)
    EXPECT_EQ(total_frames, 90);
    EXPECT_EQ(batch_count, 12);
    EXPECT_EQ(last_count, 2);

    // next() after exhaustion returns nullopt
    auto after = pipeline.next();
    EXPECT_FALSE(after.has_value());
}

// ============================================================================
// pipeline output matches direct kernel calls
// ============================================================================

TEST(PipelineTest, MatchesDirectKernels) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    const int BATCH_SIZE = 8;

    // pipeline path
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

    // preprocess with fused kernel
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
// normalize only (no resize) — output at source resolution
// ============================================================================

TEST(PipelineTest, NormalizeOnly) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    // output at source resolution (320x240)
    EXPECT_EQ(b->height(), 240);
    EXPECT_EQ(b->width(), 320);
    EXPECT_EQ(b->channels(), 3);
    EXPECT_EQ(b->count(), 4);

    // spot-check values are in ImageNet normalized range
    size_t total = static_cast<size_t>(b->count()) * 3 * 240 * 320;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < total; i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }
}

// ============================================================================
// resize only (no normalize) — values in [0, 255]
// ============================================================================

TEST(PipelineTest, ResizeOnly) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    EXPECT_EQ(b->height(), DST_H);
    EXPECT_EQ(b->width(), DST_W);
    EXPECT_EQ(b->count(), 4);

    // values should be in [0, 255] (not normalized)
    size_t total = static_cast<size_t>(b->count()) * 3 * DST_H * DST_W;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < total; i += 97) {
        EXPECT_GE(host[i], -1.0f) << "value below 0 at " << i;  // small tolerance
        EXPECT_LE(host[i], 256.0f) << "value above 255 at " << i;
    }
}

// ============================================================================
// color convert only — no resize, no normalize
// ============================================================================

TEST(PipelineTest, ColorConvertOnly) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    EXPECT_EQ(b->height(), 240);
    EXPECT_EQ(b->width(), 320);
    EXPECT_EQ(b->channels(), 3);
    EXPECT_EQ(b->count(), 4);

    // float32 RGB in [0, 255]
    size_t total = static_cast<size_t>(b->count()) * 3 * 240 * 320;
    std::vector<float> host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < total; i += 97) {
        EXPECT_GE(host[i], -1.0f) << "value below 0 at " << i;
        EXPECT_LE(host[i], 256.0f) << "value above 255 at " << i;
    }

    // non-trivial content
    float first = host[0];
    bool all_same = true;
    for (size_t i = 1; i < total; i += 31) {
        if (std::abs(host[i] - first) > 1e-5f) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same) << "batch contains all identical values";
}

// ============================================================================
// auto color matrix — BT.601 for ≤720p
// ============================================================================

TEST(PipelineTest, AutoColorMatrix) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(1)
        .build();

    auto& cfg = pipeline.config();
    // test video is 320x240, ≤720p → should auto-select BT.601
    EXPECT_FLOAT_EQ(cfg.color_matrix.r.x, cuframe::BT601.r.x);
    EXPECT_FLOAT_EQ(cfg.color_matrix.r.y, cuframe::BT601.r.y);
    EXPECT_FLOAT_EQ(cfg.color_matrix.r.z, cuframe::BT601.r.z);
    EXPECT_FLOAT_EQ(cfg.color_matrix.g.x, cuframe::BT601.g.x);
    EXPECT_FLOAT_EQ(cfg.color_matrix.g.y, cuframe::BT601.g.y);
    EXPECT_FLOAT_EQ(cfg.color_matrix.g.z, cuframe::BT601.g.z);
    EXPECT_FLOAT_EQ(cfg.color_matrix.b.x, cuframe::BT601.b.x);
    EXPECT_FLOAT_EQ(cfg.color_matrix.b.y, cuframe::BT601.b.y);
    EXPECT_FLOAT_EQ(cfg.color_matrix.b.z, cuframe::BT601.b.z);
}

// ============================================================================
// error: no input
// ============================================================================

TEST(PipelineTest, ErrorNoInput) {
    EXPECT_THROW(
        cuframe::Pipeline::builder().batch(8).build(),
        std::invalid_argument
    );
}

// ============================================================================
// error: bad file
// ============================================================================

TEST(PipelineTest, ErrorBadFile) {
    EXPECT_THROW(
        cuframe::Pipeline::builder().input("nonexistent.mp4").build(),
        std::runtime_error
    );
}

// ============================================================================
// pool backpressure — blocking acquire when pool exhausted
// ============================================================================

TEST(PipelineTest, PoolBackpressure) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .pool_size(1)
        .build();

    // grab the only pool slot
    auto first = pipeline.next();
    ASSERT_TRUE(first.has_value());

    // next() should block since pool is exhausted
    std::atomic<bool> completed{false};
    std::shared_ptr<cuframe::GpuFrameBatch> second_result;

    std::thread t([&] {
        auto result = pipeline.next();
        if (result.has_value())
            second_result = *result;
        completed.store(true);
    });

    // give the thread time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(completed.load()) << "next() should be blocked waiting for pool";

    // release the first batch — unblocks the pool
    first->reset();

    // wait for the thread to complete
    t.join();
    EXPECT_TRUE(completed.load());
    EXPECT_NE(second_result, nullptr);
}
