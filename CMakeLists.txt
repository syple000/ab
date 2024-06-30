cmake_minimum_required(VERSION 3.28.3)

project(auto_engine CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
add_compile_options(-Wno-unused-result)

add_definitions(-DGLOG_USE_GLOG_EXPORT=1)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/includes)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/libs)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/base)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op)

add_subdirectory(base)
add_subdirectory(op)

set(src_list main.cc)
add_executable(main ${src_list})
target_link_libraries(main op base glog gtest)