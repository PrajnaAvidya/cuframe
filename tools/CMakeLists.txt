add_executable(dump_frame dump_frame.cpp)
target_link_libraries(dump_frame PRIVATE cuframe)

add_executable(occupancy_report occupancy_report.cpp)
target_link_libraries(occupancy_report PRIVATE cuframe)
