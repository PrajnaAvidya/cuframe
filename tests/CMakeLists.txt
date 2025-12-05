include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_placeholder test_placeholder.cpp)
target_link_libraries(test_placeholder PRIVATE GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_placeholder)
