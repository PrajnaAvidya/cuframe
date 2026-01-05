#include "cuframe/batch_pool.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>

TEST(BatchPoolTest, Construction) {
    cuframe::BatchPool pool(3, 8, 3, 64, 64);
    EXPECT_EQ(pool.capacity(), 3);
    EXPECT_EQ(pool.available(), 3);
}

TEST(BatchPoolTest, AcquireAll) {
    cuframe::BatchPool pool(3, 4, 3, 32, 32);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    auto b2 = pool.acquire();

    ASSERT_NE(b0, nullptr);
    ASSERT_NE(b1, nullptr);
    ASSERT_NE(b2, nullptr);
    EXPECT_EQ(pool.available(), 0);

    // try_acquire should fail when exhausted
    auto b3 = pool.try_acquire();
    EXPECT_EQ(b3, nullptr);
}

TEST(BatchPoolTest, ReleaseAndReacquire) {
    cuframe::BatchPool pool(3, 4, 3, 32, 32);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    auto b2 = pool.acquire();
    EXPECT_EQ(pool.available(), 0);

    // release one
    b1.reset();
    EXPECT_EQ(pool.available(), 1);

    // reacquire succeeds
    auto b3 = pool.try_acquire();
    ASSERT_NE(b3, nullptr);
    EXPECT_EQ(pool.available(), 0);
}

TEST(BatchPoolTest, SharedPtrCopy) {
    cuframe::BatchPool pool(1, 4, 3, 32, 32);

    auto b0 = pool.acquire();
    EXPECT_EQ(pool.available(), 0);

    // copy the shared_ptr
    auto b0_copy = b0;
    EXPECT_EQ(pool.available(), 0);

    // drop original — batch still alive via copy
    b0.reset();
    EXPECT_EQ(pool.available(), 0);

    // drop copy — batch returns to pool
    b0_copy.reset();
    EXPECT_EQ(pool.available(), 1);
}

TEST(BatchPoolTest, PartialBatchCount) {
    cuframe::BatchPool pool(1, 8, 3, 32, 32);

    auto batch = pool.acquire();
    ASSERT_NE(batch, nullptr);

    // initially count == batch_size (full batch)
    EXPECT_EQ(batch->count(), batch->batch_size());
    EXPECT_EQ(batch->count(), 8);

    // set partial count
    batch->set_count(3);
    EXPECT_EQ(batch->count(), 3);
    EXPECT_EQ(batch->batch_size(), 8);  // capacity unchanged

    // set back to full
    batch->set_count(8);
    EXPECT_EQ(batch->count(), 8);

    // zero is valid (empty batch)
    batch->set_count(0);
    EXPECT_EQ(batch->count(), 0);

    // out of range throws
    EXPECT_THROW(batch->set_count(-1), std::invalid_argument);
    EXPECT_THROW(batch->set_count(9), std::invalid_argument);
}

TEST(BatchPoolTest, AcquireReleaseCycle) {
    cuframe::BatchPool pool(2, 4, 3, 32, 32);

    for (int i = 0; i < 100; ++i) {
        auto b = pool.acquire();
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->batch_size(), 4);
        // b goes out of scope, returns to pool
    }
    EXPECT_EQ(pool.available(), 2);
}

TEST(BatchPoolTest, BatchDataDistinct) {
    cuframe::BatchPool pool(3, 4, 3, 32, 32);

    auto b0 = pool.acquire();
    auto b1 = pool.acquire();
    auto b2 = pool.acquire();

    // each batch should have a distinct device pointer
    EXPECT_NE(b0->data(), b1->data());
    EXPECT_NE(b1->data(), b2->data());
    EXPECT_NE(b0->data(), b2->data());
}

TEST(BatchPoolTest, BlockingAcquire) {
    cuframe::BatchPool pool(1, 4, 3, 32, 32);

    auto b0 = pool.acquire();
    EXPECT_EQ(pool.available(), 0);

    std::atomic<bool> acquired{false};

    // launch thread that will block on acquire
    std::thread t([&pool, &acquired] {
        auto b = pool.acquire();
        acquired.store(true);
        EXPECT_NE(b, nullptr);
    });

    // give the thread time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(acquired.load());

    // release — unblocks the thread
    b0.reset();
    t.join();
    EXPECT_TRUE(acquired.load());
    EXPECT_EQ(pool.available(), 1);
}
