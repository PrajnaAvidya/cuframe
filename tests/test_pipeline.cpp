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
// BGR flag — planes swap correctly
// ============================================================================

TEST(PipelineTest, BGRFlag) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto rgb_pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(1)
        .build();

    auto bgr_pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .channel_order_bgr()
        .batch(1)
        .build();

    EXPECT_FALSE(rgb_pipeline.config().bgr);
    EXPECT_TRUE(bgr_pipeline.config().bgr);

    auto rgb_batch = rgb_pipeline.next();
    auto bgr_batch = bgr_pipeline.next();
    ASSERT_TRUE(rgb_batch.has_value());
    ASSERT_TRUE(bgr_batch.has_value());

    int plane = DST_W * DST_H;
    size_t total = 3ULL * plane;
    std::vector<float> rgb(total), bgr(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(rgb.data(), (*rgb_batch)->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));
    CUFRAME_CUDA_CHECK(cudaMemcpy(bgr.data(), (*bgr_batch)->data(),
                                   total * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < plane; i += 37) {
        EXPECT_FLOAT_EQ(bgr[0 * plane + i], rgb[2 * plane + i])
            << "BGR plane 0 != RGB plane 2 at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[1 * plane + i], rgb[1 * plane + i])
            << "G plane differs at pixel " << i;
        EXPECT_FLOAT_EQ(bgr[2 * plane + i], rgb[0 * plane + i])
            << "BGR plane 2 != RGB plane 0 at pixel " << i;
    }
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

// ============================================================================
// center crop + resize — fused path, full iteration
// ============================================================================

TEST(PipelineTest, CenterCropWithResize) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(256, 256, cuframe::ResizeMode::STRETCH)
        .center_crop(224, 224)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total_frames = 0;
    while (auto batch = pipeline.next()) {
        auto& b = *batch;
        EXPECT_EQ(b->channels(), 3);
        EXPECT_EQ(b->height(), 224);
        EXPECT_EQ(b->width(), 224);
        EXPECT_GT(b->count(), 0);
        EXPECT_LE(b->count(), 8);

        // spot-check values in normalized range
        size_t n = static_cast<size_t>(b->count()) * 3 * 224 * 224;
        std::vector<float> host(n);
        CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), b->data(),
                                       n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i += 97) {
            EXPECT_GT(host[i], -5.0f);
            EXPECT_LT(host[i], 5.0f);
        }

        total_frames += b->count();
    }

    EXPECT_EQ(total_frames, 90);
}

// ============================================================================
// center crop only — no explicit resize, crop from source resolution
// ============================================================================

TEST(PipelineTest, CenterCropOnly) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .center_crop(200, 200)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(4)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& b = *batch;
    EXPECT_EQ(b->height(), 200);
    EXPECT_EQ(b->width(), 200);
    EXPECT_EQ(b->channels(), 3);
    EXPECT_EQ(b->count(), 4);
}

// ============================================================================
// center crop config — builder stores values correctly
// ============================================================================

TEST(PipelineTest, CenterCropConfig) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(256, 256, cuframe::ResizeMode::STRETCH)
        .center_crop(224, 224)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(1)
        .build();

    auto& cfg = pipeline.config();
    EXPECT_TRUE(cfg.has_center_crop);
    EXPECT_EQ(cfg.crop_width, 224);
    EXPECT_EQ(cfg.crop_height, 224);
    EXPECT_TRUE(cfg.has_resize);
    EXPECT_EQ(cfg.resize_width, 256);
}

// ============================================================================
// center crop error — crop larger than resize target
// ============================================================================

TEST(PipelineTest, CenterCropErrorTooLarge) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    EXPECT_THROW(
        cuframe::Pipeline::builder()
            .input(TEST_VIDEO)
            .resize(256, 256, cuframe::ResizeMode::STRETCH)
            .center_crop(300, 300)
            .build(),
        std::invalid_argument
    );
}

// ============================================================================
// device selection — config stores device_id
// ============================================================================

TEST(PipelineTest, DeviceConfig) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .device(0)
        .batch(1)
        .build();

    EXPECT_EQ(pipeline.config().device_id, 0);
}

// ============================================================================
// device selection — default is 0
// ============================================================================

TEST(PipelineTest, DeviceDefault) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(1)
        .build();

    EXPECT_EQ(pipeline.config().device_id, 0);
}

// ============================================================================
// device selection — invalid device throws
// ============================================================================

TEST(PipelineTest, DeviceInvalid) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    EXPECT_THROW(
        cuframe::Pipeline::builder()
            .input(TEST_VIDEO)
            .device(999)
            .batch(1)
            .build(),
        std::runtime_error
    );
}

// ============================================================================
// multi-GPU — pipeline on device 1 (skipped if < 2 GPUs)
// ============================================================================

TEST(PipelineTest, MultiGPU) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) GTEST_SKIP() << "need >= 2 GPUs";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(320, 320)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .device(1)
        .batch(4)
        .build();

    EXPECT_EQ(pipeline.config().device_id, 1);

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ((*batch)->count(), 4);

    // verify data is accessible from device 1
    size_t n = 4ULL * 3 * 320 * 320;
    std::vector<float> host(n);
    CUFRAME_CUDA_CHECK(cudaSetDevice(1));
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), (*batch)->data(),
                                   n * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; i += 97) {
        EXPECT_GT(host[i], -5.0f);
        EXPECT_LT(host[i], 5.0f);
    }
}

// ============================================================================
// retain_decoded — retained frame has valid metadata
// ============================================================================

TEST(PipelineTest, RetainDecoded_HasValidData) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .retain_decoded(true)
        .batch(1)
        .build();

    EXPECT_TRUE(pipeline.config().retain_decoded);

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    EXPECT_EQ(pipeline.retained_count(), 1);
    auto& frame = pipeline.retained_frame(0);
    EXPECT_NE(frame.data, nullptr);
    // test video is 320x240
    EXPECT_EQ(frame.width, 320);
    EXPECT_EQ(frame.height, 240);
    EXPECT_GT(frame.pitch, 0u);
}

// ============================================================================
// retain_decoded — NV12 content is valid
// ============================================================================

TEST(PipelineTest, RetainDecoded_NV12ContentValid) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .retain_decoded(true)
        .batch(1)
        .build();

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());

    auto& frame = pipeline.retained_frame(0);
    unsigned int luma_h = frame.height;
    unsigned int chroma_h = (frame.height + 1) / 2;
    size_t nv12_bytes = frame.pitch * (luma_h + chroma_h);

    std::vector<uint8_t> host(nv12_bytes);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), frame.data,
                                   nv12_bytes, cudaMemcpyDeviceToHost));

    // luma values should be in [0, 255] and not all zero
    bool all_zero = true;
    for (size_t i = 0; i < luma_h * frame.pitch; i += 37) {
        if (host[i] != 0) all_zero = false;
    }
    EXPECT_FALSE(all_zero) << "NV12 luma plane is all zeros";
}

// ============================================================================
// retain_decoded — content changes across batches
// ============================================================================

TEST(PipelineTest, RetainDecoded_ContentChanges) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .retain_decoded(true)
        .batch(1)
        .build();

    // read first batch, snapshot full retained NV12
    auto batch1 = pipeline.next();
    ASSERT_TRUE(batch1.has_value());
    auto& f1 = pipeline.retained_frame(0);
    unsigned int luma_h = f1.height;
    unsigned int chroma_h = (f1.height + 1) / 2;
    size_t frame_bytes = f1.pitch * (luma_h + chroma_h);
    std::vector<uint8_t> snap1(frame_bytes);
    CUFRAME_CUDA_CHECK(cudaMemcpy(snap1.data(), f1.data,
                                   frame_bytes, cudaMemcpyDeviceToHost));

    // skip several frames to ensure content differs
    for (int i = 0; i < 15; ++i) pipeline.next();

    // read another batch
    auto batch2 = pipeline.next();
    ASSERT_TRUE(batch2.has_value());
    auto& f2 = pipeline.retained_frame(0);
    std::vector<uint8_t> snap2(frame_bytes);
    CUFRAME_CUDA_CHECK(cudaMemcpy(snap2.data(), f2.data,
                                   frame_bytes, cudaMemcpyDeviceToHost));

    // retained buffer pointer is reused (same alloc)
    EXPECT_EQ(f1.data, f2.data);
    // content should differ somewhere (different frame — testsrc has a frame counter)
    int diff_count = 0;
    for (size_t i = 0; i < frame_bytes; i += 1)
        if (snap1[i] != snap2[i]) ++diff_count;
    EXPECT_GT(diff_count, 0) << "retained NV12 content should change between batches";
}

// ============================================================================
// retain_decoded — disabled by default
// ============================================================================

TEST(PipelineTest, RetainDecoded_DisabledByDefault) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(1)
        .build();

    EXPECT_FALSE(pipeline.config().retain_decoded);

    auto batch = pipeline.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(pipeline.retained_count(), 0);
}

// ============================================================================
// source video metadata accessors
// ============================================================================

TEST(PipelineTest, SourceMetadata) {
    if (!test_video_exists()) GTEST_SKIP() << "test video not found";

    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(DST_W, DST_H)
        .batch(1)
        .build();

    // test video: 320x240, 30fps, 90 frames
    EXPECT_EQ(pipeline.source_width(), 320);
    EXPECT_EQ(pipeline.source_height(), 240);
    EXPECT_DOUBLE_EQ(pipeline.fps(), 30.0);
    EXPECT_EQ(pipeline.frame_count(), 90);
}
