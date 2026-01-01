add_library(cuframe STATIC
    cuframe/cuda_utils.cpp
    cuframe/device_buffer.cu
    cuframe/demuxer.cpp
    cuframe/decoder.cpp
    cuframe/frame_pool.cpp
    cuframe/gpu_frame_batch.cpp
    cuframe/kernels/color_convert.cu
    cuframe/kernels/resize.cu
    cuframe/kernels/normalize.cu
)

target_include_directories(cuframe PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# nvdec api headers (vendored from nvidia video codec sdk)
add_library(nv_codec_headers INTERFACE)
target_include_directories(nv_codec_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/nv-codec-headers/include
)

# nvdec runtime (ships with nvidia driver)
find_library(NVCUVID_LIB nvcuvid REQUIRED)

target_link_libraries(cuframe PUBLIC
    nv_codec_headers
    PkgConfig::LIBAV
    ${NVCUVID_LIB}
    CUDA::cuda_driver
    CUDA::cudart
)

add_executable(cuframe_main main.cpp)
target_link_libraries(cuframe_main PRIVATE cuframe)
