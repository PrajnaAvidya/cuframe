cmake_minimum_required(VERSION 3.24)
project(cuframe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# glibc 2.42+ (fedora 43) declares c23 math functions (cospi, sinpi, rsqrt)
# with noexcept(true), conflicting with cuda 12.9's declarations.
# -U_GNU_SOURCE prevents glibc from exposing these during cuda compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -U_GNU_SOURCE")

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
