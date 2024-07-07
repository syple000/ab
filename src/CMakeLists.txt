add_subdirectory(tensor)
add_subdirectory(op)
add_subdirectory(utils)
add_subdirectory(cuda)

# 聚合所有lib并处理include目录
target_link_libraries(
    lop
    PUBLIC
    ltensor 
    lutils 
    lcuda)
target_include_directories(lop PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(main main.cc)
target_link_libraries(main lop GTest::gtest glog::glog)