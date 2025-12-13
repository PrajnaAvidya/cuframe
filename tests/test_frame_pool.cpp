#include "cuframe/frame_pool.h"
#include <gtest/gtest.h>

TEST(FramePool, AcquireAll) {
    cuframe::FramePool pool(1024, 4);
    EXPECT_EQ(pool.available(), 4);
    EXPECT_EQ(pool.capacity(), 4);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    auto b2 = pool.acquire();
    auto b3 = pool.acquire();

    EXPECT_NE(b0, nullptr);
    EXPECT_NE(b1, nullptr);
    EXPECT_NE(b2, nullptr);
    EXPECT_NE(b3, nullptr);
    EXPECT_EQ(pool.available(), 0);

    // 5th acquire returns null
    auto b4 = pool.acquire();
    EXPECT_EQ(b4, nullptr);
}

TEST(FramePool, ReleaseAndReacquire) {
    cuframe::FramePool pool(1024, 2);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    EXPECT_EQ(pool.available(), 0);

    void* ptr = b0->data();
    b0.reset();  // return to pool
    EXPECT_EQ(pool.available(), 1);

    auto b2 = pool.acquire();
    EXPECT_NE(b2, nullptr);
    EXPECT_EQ(b2->data(), ptr);  // same buffer recycled
    EXPECT_EQ(pool.available(), 0);
}

TEST(FramePool, AutoReturn) {
    cuframe::FramePool pool(1024, 3);

    {
        auto b0 = pool.acquire();
        auto b1 = pool.acquire();
        EXPECT_EQ(pool.available(), 1);
    }
    // both went out of scope — returned to pool
    EXPECT_EQ(pool.available(), 3);
}

TEST(FramePool, BufferSize) {
    const size_t sz = 4096;
    cuframe::FramePool pool(sz, 3);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    auto b2 = pool.acquire();

    EXPECT_EQ(b0->size(), sz);
    EXPECT_EQ(b1->size(), sz);
    EXPECT_EQ(b2->size(), sz);
}

TEST(FramePool, AvailableCount) {
    cuframe::FramePool pool(512, 4);
    EXPECT_EQ(pool.available(), 4);
    EXPECT_EQ(pool.capacity(), 4);

    auto b0 = pool.acquire();
    EXPECT_EQ(pool.available(), 3);

    auto b1 = pool.acquire();
    EXPECT_EQ(pool.available(), 2);

    b0.reset();
    EXPECT_EQ(pool.available(), 3);

    b1.reset();
    EXPECT_EQ(pool.available(), 4);

    // capacity unchanged throughout
    EXPECT_EQ(pool.capacity(), 4);
}
