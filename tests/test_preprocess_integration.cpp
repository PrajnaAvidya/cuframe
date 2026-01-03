#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"
#include "cuframe/decoder.h"
#include "cuframe/demuxer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <memory>
#include <algorithm>

extern "C" {
#include <libavcodec/packet.h>
}

static const char* TEST_VIDEO = "tests/data/test_h264.mp4";
static const int DST_W = 640;
static const int DST_H = 640;
static const int BATCH_SIZE = 8;

static bool test_video_exists() {
    return std::filesystem::exists(TEST_VIDEO);
}

// fixture: decode multiple frames for full pipeline tests
class PreprocessIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_video_exists()) GTEST_SKIP() << "test video not found";
        demuxer_ = std::make_unique<cuframe::Demuxer>(TEST_VIDEO);
        decoder_ = std::make_unique<cuframe::Decoder>(demuxer_->video_info(), 20);
        decode_n_frames(BATCH_SIZE);
    }

    void decode_n_frames(int n) {
        AVPacket* pkt = av_packet_alloc();
        while (demuxer_->read_packet(pkt)) {
            decoder_->decode(pkt, frames_);
            av_packet_unref(pkt);
            if (static_cast<int>(frames_.size()) >= n) break;
        }
        if (static_cast<int>(frames_.size()) < n)
            decoder_->flush(frames_);
        av_packet_free(&pkt);
        ASSERT_GE(static_cast<int>(frames_.size()), 1) << "failed to decode any frames";
    }

    // run separate pipeline for one frame, return device pointer to normalized output.
    // caller must cudaFree the returned pointer.
    float* preprocess_separate(const cuframe::DecodedFrame& frame,
                                const cuframe::ResizeParams& resize_params) {
        int w = frame.width;
        int h = frame.height;
        unsigned int pitch = frame.pitch;
        auto* nv12 = static_cast<const uint8_t*>(frame.buffer->data());

        size_t rgb_bytes = 3 * w * h * sizeof(float);
        size_t out_bytes = 3 * DST_W * DST_H * sizeof(float);

        float* rgb_d = nullptr;
        float* resized_d = nullptr;
        CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_d, rgb_bytes));
        CUFRAME_CUDA_CHECK(cudaMalloc(&resized_d, out_bytes));

        cuframe::nv12_to_rgb_planar(nv12, rgb_d, w, h, pitch, cuframe::BT601);
        cuframe::resize_bilinear(rgb_d, resized_d, resize_params);
        cuframe::normalize(resized_d, resized_d, DST_W, DST_H, cuframe::IMAGENET_NORM);

        cudaFree(rgb_d);
        return resized_d;
    }

    // run fused pipeline for one frame, return device pointer.
    // caller must cudaFree the returned pointer.
    float* preprocess_fused(const cuframe::DecodedFrame& frame,
                             const cuframe::ResizeParams& resize_params) {
        auto* nv12 = static_cast<const uint8_t*>(frame.buffer->data());
        size_t out_bytes = 3 * DST_W * DST_H * sizeof(float);

        float* out_d = nullptr;
        CUFRAME_CUDA_CHECK(cudaMalloc(&out_d, out_bytes));

        cuframe::fused_nv12_to_tensor(nv12, out_d,
                                       frame.width, frame.height, frame.pitch,
                                       resize_params, cuframe::BT601, cuframe::IMAGENET_NORM);
        return out_d;
    }

    std::unique_ptr<cuframe::Demuxer> demuxer_;
    std::unique_ptr<cuframe::Decoder> decoder_;
    std::vector<cuframe::DecodedFrame> frames_;
};

TEST_F(PreprocessIntegrationTest, SeparatePipelineEndToEnd) {
    int n = std::min(BATCH_SIZE, static_cast<int>(frames_.size()));
    int w = frames_[0].width;
    int h = frames_[0].height;
    auto resize_params = cuframe::make_letterbox_params(w, h, DST_W, DST_H);

    // preprocess each frame
    std::vector<float*> frame_ptrs;
    for (int i = 0; i < n; ++i)
        frame_ptrs.push_back(preprocess_separate(frames_[i], resize_params));

    // batch
    cuframe::GpuFrameBatch batch(n, 3, DST_H, DST_W);
    std::vector<const float*> const_ptrs(frame_ptrs.begin(), frame_ptrs.end());
    cuframe::batch_frames(batch, const_ptrs.data(), n);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // verify dimensions
    EXPECT_EQ(batch.batch_size(), n);
    EXPECT_EQ(batch.channels(), 3);
    EXPECT_EQ(batch.height(), DST_H);
    EXPECT_EQ(batch.width(), DST_W);

    // copy to host and verify values
    std::vector<float> host(n * 3 * DST_H * DST_W);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), batch.data(),
                                   batch.total_size_bytes(), cudaMemcpyDeviceToHost));

    // ImageNet-normalized values should be roughly in [-3, 3]
    for (size_t i = 0; i < host.size(); i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    // verify non-trivial content (not all same value)
    float first = host[0];
    bool all_same = true;
    for (size_t i = 1; i < host.size(); i += 31) {
        if (std::abs(host[i] - first) > 1e-5f) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same) << "batch contains all identical values";

    for (auto* p : frame_ptrs) cudaFree(p);
}

TEST_F(PreprocessIntegrationTest, FusedPipelineEndToEnd) {
    int n = std::min(BATCH_SIZE, static_cast<int>(frames_.size()));
    int w = frames_[0].width;
    int h = frames_[0].height;
    auto resize_params = cuframe::make_letterbox_params(w, h, DST_W, DST_H);

    // preprocess each frame with fused kernel
    std::vector<float*> frame_ptrs;
    for (int i = 0; i < n; ++i)
        frame_ptrs.push_back(preprocess_fused(frames_[i], resize_params));

    // batch
    cuframe::GpuFrameBatch batch(n, 3, DST_H, DST_W);
    std::vector<const float*> const_ptrs(frame_ptrs.begin(), frame_ptrs.end());
    cuframe::batch_frames(batch, const_ptrs.data(), n);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(batch.batch_size(), n);

    // verify values
    std::vector<float> host(n * 3 * DST_H * DST_W);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), batch.data(),
                                   batch.total_size_bytes(), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host.size(); i += 97) {
        EXPECT_GT(host[i], -5.0f) << "value too negative at " << i;
        EXPECT_LT(host[i], 5.0f) << "value too large at " << i;
    }

    for (auto* p : frame_ptrs) cudaFree(p);
}

TEST_F(PreprocessIntegrationTest, SeparateVsFusedBatchEquivalence) {
    int n = std::min(BATCH_SIZE, static_cast<int>(frames_.size()));
    int w = frames_[0].width;
    int h = frames_[0].height;
    auto resize_params = cuframe::make_letterbox_params(w, h, DST_W, DST_H);

    // process all frames through both paths
    std::vector<float*> sep_ptrs, fused_ptrs;
    for (int i = 0; i < n; ++i) {
        sep_ptrs.push_back(preprocess_separate(frames_[i], resize_params));
        fused_ptrs.push_back(preprocess_fused(frames_[i], resize_params));
    }

    // batch both
    cuframe::GpuFrameBatch sep_batch(n, 3, DST_H, DST_W);
    cuframe::GpuFrameBatch fused_batch(n, 3, DST_H, DST_W);

    std::vector<const float*> sep_const(sep_ptrs.begin(), sep_ptrs.end());
    std::vector<const float*> fused_const(fused_ptrs.begin(), fused_ptrs.end());

    cuframe::batch_frames(sep_batch, sep_const.data(), n);
    cuframe::batch_frames(fused_batch, fused_const.data(), n);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // copy to host and compare
    size_t total = n * 3 * DST_H * DST_W;
    std::vector<float> sep_host(total), fused_host(total);
    CUFRAME_CUDA_CHECK(cudaMemcpy(sep_host.data(), sep_batch.data(),
                                   sep_batch.total_size_bytes(), cudaMemcpyDeviceToHost));
    CUFRAME_CUDA_CHECK(cudaMemcpy(fused_host.data(), fused_batch.data(),
                                   fused_batch.total_size_bytes(), cudaMemcpyDeviceToHost));

    int mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        float diff = std::abs(sep_host[i] - fused_host[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) ++mismatch_count;
    }

    EXPECT_EQ(mismatch_count, 0)
        << mismatch_count << " mismatches out of " << total
        << " (" << n << " frames), max diff = " << max_diff;

    for (auto* p : sep_ptrs) cudaFree(p);
    for (auto* p : fused_ptrs) cudaFree(p);
}

TEST_F(PreprocessIntegrationTest, DumpPreNormFrame) {
    auto& frame = frames_[0];
    int w = frame.width;
    int h = frame.height;
    auto* nv12 = static_cast<const uint8_t*>(frame.buffer->data());
    auto resize_params = cuframe::make_letterbox_params(w, h, DST_W, DST_H);

    // color convert + resize (no normalize)
    size_t rgb_bytes = 3 * w * h * sizeof(float);
    size_t out_bytes = 3 * DST_W * DST_H * sizeof(float);

    float* rgb_d = nullptr;
    float* resized_d = nullptr;
    CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_d, rgb_bytes));
    CUFRAME_CUDA_CHECK(cudaMalloc(&resized_d, out_bytes));

    cuframe::nv12_to_rgb_planar(nv12, rgb_d, w, h, frame.pitch, cuframe::BT601);
    cuframe::resize_bilinear(rgb_d, resized_d, resize_params);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // copy to host
    std::vector<float> host(3 * DST_W * DST_H);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), resized_d, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(rgb_d);
    cudaFree(resized_d);

    // convert planar float32 [0,255] → packed uint8 RGB for PPM
    std::vector<uint8_t> packed(DST_W * DST_H * 3);
    int plane = DST_W * DST_H;
    for (int y = 0; y < DST_H; ++y) {
        for (int x = 0; x < DST_W; ++x) {
            int pixel = y * DST_W + x;
            int out_idx = pixel * 3;
            for (int c = 0; c < 3; ++c) {
                float v = host[c * plane + pixel];
                v = std::max(0.0f, std::min(255.0f, v));
                packed[out_idx + c] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    }

    // write PPM P6
    std::string dump_path = "tests/data/preprocess_dump.ppm";
    std::ofstream out(dump_path, std::ios::binary);
    ASSERT_TRUE(out.is_open()) << "failed to open " << dump_path;

    std::string header = "P6\n" + std::to_string(DST_W) + " " + std::to_string(DST_H) + "\n255\n";
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(packed.data()), packed.size());
    out.close();

    // verify file exists with expected size
    auto file_size = std::filesystem::file_size(dump_path);
    size_t expected = header.size() + DST_W * DST_H * 3;
    EXPECT_EQ(file_size, expected);

    printf("preprocessed frame dumped to %s\n", dump_path.c_str());
    printf("view with: display %s\n", dump_path.c_str());
}
