include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_device_buffer test_device_buffer.cpp)
target_link_libraries(test_device_buffer PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_device_buffer)

add_executable(test_demuxer test_demuxer.cpp)
target_link_libraries(test_demuxer PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_demuxer WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_decoder test_decoder.cpp)
target_link_libraries(test_decoder PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_decoder WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_frame_pool test_frame_pool.cpp)
target_link_libraries(test_frame_pool PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_frame_pool)

add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_integration WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_color_convert test_color_convert.cpp)
target_link_libraries(test_color_convert PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_color_convert WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_resize test_resize.cpp)
target_link_libraries(test_resize PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_resize)

add_executable(test_normalize test_normalize.cpp)
target_link_libraries(test_normalize PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_normalize)

add_executable(test_gpu_frame_batch test_gpu_frame_batch.cpp)
target_link_libraries(test_gpu_frame_batch PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_gpu_frame_batch)

add_executable(test_fused_preprocess test_fused_preprocess.cpp)
target_link_libraries(test_fused_preprocess PRIVATE cuframe GTest::gtest_main)
gtest_discover_tests(test_fused_preprocess WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
