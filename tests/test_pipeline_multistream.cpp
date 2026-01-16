#include "cuframe/pipeline.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <chrono>

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
