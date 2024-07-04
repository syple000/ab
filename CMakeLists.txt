cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
project(auto_engine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
add_compile_options(-Wno-unused-result)

enable_language(CUDA)
message(STATUS "cuda include_dir: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "cude link_dir: " ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)

add_subdirectory(src)

set(src_list main.cc)
add_executable(main ${src_list})

target_link_libraries(main PRIVATE lop ltensor lcuda lutils glog::glog Folly::folly Folly::folly_deps GTest::gtest)