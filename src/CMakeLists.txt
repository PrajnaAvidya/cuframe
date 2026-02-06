add_library(cuframe STATIC
    cuframe/cuda_utils.cpp
    cuframe/device_buffer.cu
    cuframe/demuxer.cpp
    cuframe/decoder.cpp
    cuframe/frame_pool.cpp
    cuframe/gpu_frame_batch.cpp
    cuframe/batch_pool.cpp
    cuframe/pipeline.cpp
    cuframe/kernels/color_convert.cu
    cuframe/kernels/resize.cu
    cuframe/kernels/normalize.cu
    cuframe/kernels/fused_preprocess.cu
    cuframe/kernels/roi_crop.cu
)

set_target_properties(cuframe PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

if(CUFRAME_PYTHON_BINDINGS)
    nanobind_add_module(_cuframe ${PROJECT_SOURCE_DIR}/python/bindings.cpp)
    target_link_libraries(_cuframe PRIVATE cuframe)
endif()
