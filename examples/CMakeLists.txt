add_executable(basic_pipeline basic_pipeline.cpp)
target_link_libraries(basic_pipeline PRIVATE cuframe)

# inference examples are illustrative — compile but framework calls are pseudocode
add_executable(tensorrt_inference tensorrt_inference.cpp)
target_link_libraries(tensorrt_inference PRIVATE cuframe)

add_executable(onnxruntime_inference onnxruntime_inference.cpp)
target_link_libraries(onnxruntime_inference PRIVATE cuframe)

add_executable(classification_pipeline classification_pipeline.cpp)
target_link_libraries(classification_pipeline PRIVATE cuframe)

add_executable(two_stage_pipeline two_stage_pipeline.cpp)
target_link_libraries(two_stage_pipeline PRIVATE cuframe)

add_executable(temporal_pipeline temporal_pipeline.cpp)
target_link_libraries(temporal_pipeline PRIVATE cuframe)

add_executable(error_recovery error_recovery.cpp)
target_link_libraries(error_recovery PRIVATE cuframe)
