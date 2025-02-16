add_executable(quant_benchmark bench.cpp)
target_include_directories(quant_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(quant_benchmark PRIVATE quant)