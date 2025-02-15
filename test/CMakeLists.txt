add_executable(quant_test test.cpp)
target_include_directories(quant_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(quant_test PRIVATE quant_kernels)