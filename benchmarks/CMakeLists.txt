add_executable(bench_decode bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE cuframe)

# swscale for cpu baseline benchmark only — not linked to cuframe library
pkg_check_modules(LIBSWSCALE REQUIRED IMPORTED_TARGET libswscale)

add_executable(bench_preprocess bench_preprocess.cpp)
target_link_libraries(bench_preprocess PRIVATE cuframe PkgConfig::LIBSWSCALE)

add_executable(bench_pipeline bench_pipeline.cpp)
target_link_libraries(bench_pipeline PRIVATE cuframe)
