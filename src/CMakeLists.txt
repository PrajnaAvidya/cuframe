# interface-only until we add real sources in chunk 3
add_library(cuframe INTERFACE)

target_include_directories(cuframe INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

# nvdec api headers (vendored from nvidia video codec sdk)
add_library(nv_codec_headers INTERFACE)
target_include_directories(nv_codec_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/nv-codec-headers/include
)

# nvdec runtime (ships with nvidia driver)
find_library(NVCUVID_LIB nvcuvid REQUIRED)

target_link_libraries(cuframe INTERFACE
    nv_codec_headers
    PkgConfig::LIBAV
    ${NVCUVID_LIB}
    CUDA::cuda_driver
    CUDA::cudart
)

add_executable(cuframe_main main.cpp)
target_link_libraries(cuframe_main PRIVATE cuframe)
