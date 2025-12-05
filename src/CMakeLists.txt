# interface-only until we add real sources in chunk 3
add_library(cuframe INTERFACE)

target_include_directories(cuframe INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(cuframe_main main.cpp)
target_link_libraries(cuframe_main PRIVATE CUDA::cudart)
