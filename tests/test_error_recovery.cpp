#include "cuframe/pipeline.h"
#include "cuframe/cuda_utils.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";
static const char* CORRUPT_VIDEO = "tests/data/test_corrupt.mp4";
static const char* HEAVY_CORRUPT_VIDEO = "tests/data/test_heavy_corrupt.mp4";

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

class ErrorRecoveryTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found";
        create_test_files();
    }

    void TearDown() override {
        std::filesystem::remove(CORRUPT_VIDEO);
        std::filesystem::remove(HEAVY_CORRUPT_VIDEO);
    }

    void create_test_files() {
        auto size = std::filesystem::file_size(TEST_VIDEO);
        std::ifstream in(TEST_VIDEO, std::ios::binary);
        std::vector<char> data(size);
        in.read(data.data(), static_cast<std::streamsize>(size));

        // light corruption: 1KB of garbage at ~1/3 of file.
        // container metadata (moov) stays intact so the file opens,
        // but packet data is corrupted mid-stream.
        {
            auto corrupt = data;
            size_t offset = size / 3;
            for (size_t i = 0; i < 1024 && (offset + i) < size; ++i)
                corrupt[offset + i] = static_cast<char>(0xFF);
            std::ofstream out(CORRUPT_VIDEO, std::ios::binary);
            out.write(corrupt.data(), static_cast<std::streamsize>(size));
        }

        // heavy corruption: wipe 40% of the mdat payload.
        // keeps moov intact but destroys a large chunk of video data.
        {
            auto corrupt = data;
            size_t start = size / 4;
            size_t len = size * 2 / 5;
            for (size_t i = 0; i < len && (start + i) < size; ++i)
                corrupt[start + i] = static_cast<char>(0xFF);
            std::ofstream out(HEAVY_CORRUPT_VIDEO, std::ios::binary);
            out.write(corrupt.data(), static_cast<std::streamsize>(size));
        }
    }
};

// default (THROW) — corrupt file throws during iteration
TEST_F(ErrorRecoveryTest, ThrowPolicy_CorruptThrows) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(HEAVY_CORRUPT_VIDEO)
        .resize(640, 640)
        .batch(8)
        .build();

    EXPECT_THROW({
        while (pipeline.next()) {}
    }, std::runtime_error);
}

// SKIP — heavily corrupt file produces partial output, no exception
TEST_F(ErrorRecoveryTest, SkipPolicy_CorruptPartial) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(HEAVY_CORRUPT_VIDEO)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .build();

    int total = 0;
    while (auto batch = pipeline.next())
        total += (*batch)->count();

    // should get some frames but fewer than the full 90
    EXPECT_GT(total, 0);
    EXPECT_LT(total, 90);
    EXPECT_GT(pipeline.error_count(), 0u);
}

// SKIP — callback fires and count matches
TEST_F(ErrorRecoveryTest, SkipPolicy_CallbackFires) {
    std::vector<std::string> errors;

    auto pipeline = cuframe::Pipeline::builder()
        .input(HEAVY_CORRUPT_VIDEO)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .on_error([&](const cuframe::ErrorInfo& info) {
            errors.push_back(info.message);
        })
        .build();

    while (pipeline.next()) {}

    EXPECT_FALSE(errors.empty());
    EXPECT_EQ(errors.size(), pipeline.error_count());
}

// SKIP — valid file produces zero errors, same output as THROW
TEST_F(ErrorRecoveryTest, SkipPolicy_ValidFileZeroErrors) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .resize(640, 640)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .build();

    int total = 0;
    while (auto batch = pipeline.next())
        total += (*batch)->count();

    EXPECT_EQ(total, 90);
    EXPECT_EQ(pipeline.error_count(), 0u);
}

// SKIP — no callback set, still works
TEST_F(ErrorRecoveryTest, SkipPolicy_NoCallback) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(HEAVY_CORRUPT_VIDEO)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .build();

    while (pipeline.next()) {}
    EXPECT_GT(pipeline.error_count(), 0u);
}

// error_count starts at zero
TEST_F(ErrorRecoveryTest, ErrorCountStartsZero) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(1)
        .build();

    EXPECT_EQ(pipeline.error_count(), 0u);
}

// light corruption — recovers after small corrupt section
TEST_F(ErrorRecoveryTest, SkipPolicy_LightCorruption) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(CORRUPT_VIDEO)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .build();

    int total = 0;
    while (auto batch = pipeline.next())
        total += (*batch)->count();

    // should get most frames — small corruption zone
    EXPECT_GT(total, 0);
}

// default policy is THROW
TEST_F(ErrorRecoveryTest, DefaultPolicyIsThrow) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(TEST_VIDEO)
        .batch(1)
        .build();

    EXPECT_EQ(pipeline.config().error_policy, cuframe::ErrorPolicy::THROW);
}
