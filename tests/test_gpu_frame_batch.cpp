#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

TEST(GpuFrameBatchTest, Construction) {
    cuframe::GpuFrameBatch batch(4, 3, 640, 640);
    EXPECT_EQ(batch.batch_size(), 4);
    EXPECT_EQ(batch.channels(), 3);
    EXPECT_EQ(batch.height(), 640);
    EXPECT_EQ(batch.width(), 640);
    EXPECT_NE(batch.data(), nullptr);
    EXPECT_EQ(batch.frame_size_bytes(), 3 * 640 * 640 * sizeof(float));
    EXPECT_EQ(batch.total_size_bytes(), 4 * 3 * 640 * 640 * sizeof(float));
    EXPECT_EQ(batch.count(), 0);
}

TEST(GpuFrameBatchTest, FrameOffsets) {
    cuframe::GpuFrameBatch batch(4, 3, 640, 640);
    const size_t frame_elems = 3 * 640 * 640;

    EXPECT_EQ(batch.frame(0), batch.data());
    EXPECT_EQ(batch.frame(1), batch.data() + frame_elems);
    EXPECT_EQ(batch.frame(2), batch.data() + 2 * frame_elems);
    EXPECT_EQ(batch.frame(3), batch.data() + 3 * frame_elems);
}

TEST(GpuFrameBatchTest, BatchFramesRoundTrip) {
    const int n = 4, c = 3, h = 4, w = 4;
    const size_t frame_elems = c * h * w;
    const size_t frame_bytes = frame_elems * sizeof(float);

    // create per-frame device buffers with distinct patterns
    std::vector<float*> d_frames(n);
    for (int i = 0; i < n; ++i) {
        CUFRAME_CUDA_CHECK(cudaMalloc(&d_frames[i], frame_bytes));
        std::vector<float> pattern(frame_elems, static_cast<float>(i + 1) * 10.0f);
        CUFRAME_CUDA_CHECK(cudaMemcpy(d_frames[i], pattern.data(), frame_bytes,
                                       cudaMemcpyHostToDevice));
    }

    // batch them
    cuframe::GpuFrameBatch batch(n, c, h, w);
    std::vector<const float*> ptrs(d_frames.begin(), d_frames.end());
    cuframe::batch_frames(batch, ptrs.data(), n);
    CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    // download entire batch and verify
    std::vector<float> host(n * frame_elems);
    CUFRAME_CUDA_CHECK(cudaMemcpy(host.data(), batch.data(),
                                   batch.total_size_bytes(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        float expected = static_cast<float>(i + 1) * 10.0f;
        for (size_t j = 0; j < frame_elems; ++j) {
            EXPECT_FLOAT_EQ(host[i * frame_elems + j], expected)
                << "frame=" << i << " elem=" << j;
        }
    }

    for (int i = 0; i < n; ++i) cudaFree(d_frames[i]);
}

TEST(GpuFrameBatchTest, MoveConstruct) {
    cuframe::GpuFrameBatch a(2, 3, 32, 32);
    float* orig_data = a.data();
    ASSERT_NE(orig_data, nullptr);

    cuframe::GpuFrameBatch b(std::move(a));
    EXPECT_EQ(b.data(), orig_data);
    EXPECT_EQ(b.batch_size(), 2);
    EXPECT_EQ(b.channels(), 3);
    EXPECT_EQ(b.height(), 32);
    EXPECT_EQ(b.width(), 32);
    EXPECT_EQ(a.data(), nullptr);
}

TEST(GpuFrameBatchTest, BatchFramesOverflowThrows) {
    const int c = 3, h = 4, w = 4;
    const size_t frame_bytes = c * h * w * sizeof(float);

    cuframe::GpuFrameBatch batch(2, c, h, w);

    // allocate 3 frame pointers but batch only holds 2
    std::vector<float*> d_frames(3);
    for (int i = 0; i < 3; ++i)
        CUFRAME_CUDA_CHECK(cudaMalloc(&d_frames[i], frame_bytes));

    std::vector<const float*> ptrs(d_frames.begin(), d_frames.end());
    EXPECT_THROW(cuframe::batch_frames(batch, ptrs.data(), 3), std::invalid_argument);

    for (int i = 0; i < 3; ++i) cudaFree(d_frames[i]);
}

TEST(GpuFrameBatchTest, StreamExecution) {
    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    const int n = 2, c = 3, h = 8, w = 8;
    const size_t frame_elems = c * h * w;
    const size_t frame_bytes = frame_elems * sizeof(float);

    std::vector<float*> d_frames(n);
    for (int i = 0; i < n; ++i) {
        CUFRAME_CUDA_CHECK(cudaMalloc(&d_frames[i], frame_bytes));
        std::vector<float> pattern(frame_elems, static_cast<float>(i + 1) * 5.0f);
        CUFRAME_CUDA_CHECK(cudaMemcpy(d_frames[i], pattern.data(), frame_bytes,
                                       cudaMemcpyHostToDevice));
    }

    cuframe::GpuFrameBatch batch(n, c, h, w);
    std::vector<const float*> ptrs(d_frames.begin(), d_frames.end());
    cuframe::batch_frames(batch, ptrs.data(), n, stream);
    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));

    // verify first element of each frame
    for (int i = 0; i < n; ++i) {
        float val = 0.0f;
        CUFRAME_CUDA_CHECK(cudaMemcpy(&val, batch.frame(i), sizeof(float),
                                       cudaMemcpyDeviceToHost));
        EXPECT_FLOAT_EQ(val, static_cast<float>(i + 1) * 5.0f);
    }

    for (int i = 0; i < n; ++i) cudaFree(d_frames[i]);
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));
}
