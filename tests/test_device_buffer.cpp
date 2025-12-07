#include "cuframe/device_buffer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdexcept>

TEST(DeviceBuffer, AllocateAndFree) {
    cuframe::DeviceBuffer buf(1024 * 1024); // 1 MB
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024 * 1024);
}

TEST(DeviceBuffer, MoveConstruct) {
    cuframe::DeviceBuffer a(1024);
    void* orig_ptr = a.data();
    ASSERT_NE(orig_ptr, nullptr);

    cuframe::DeviceBuffer b(std::move(a));
    EXPECT_EQ(b.data(), orig_ptr);
    EXPECT_EQ(b.size(), 1024);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(DeviceBuffer, MoveAssign) {
    cuframe::DeviceBuffer a(1024);
    cuframe::DeviceBuffer b(2048);
    void* ptr_a = a.data();

    b = std::move(a);
    EXPECT_EQ(b.data(), ptr_a);
    EXPECT_EQ(b.size(), 1024);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(DeviceBuffer, ZeroSize) {
    cuframe::DeviceBuffer buf(0);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0);
}

TEST(DeviceBuffer, AbsurdSizeThrows) {
    // ~1 TB, way beyond any GPU
    EXPECT_THROW(cuframe::DeviceBuffer(1ULL << 40), std::runtime_error);
}

TEST(DeviceBuffer, StreamOrdered) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        cuframe::DeviceBuffer buf(4096, stream);
        EXPECT_NE(buf.data(), nullptr);
        EXPECT_EQ(buf.size(), 4096);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
