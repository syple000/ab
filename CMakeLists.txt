cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
project(auto_engine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
add_compile_options(-Wno-unused-result)

# clangd对cuda输出的commands.json支持不好，想要正确识别，请删除：/usr/share/cmake-3.28/Modules/Compiler/NVIDIA.cmake
# 中RESPONSE_FILE & options-file相关内容
enable_language(CUDA)

find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

set(src_list main.cc)
add_executable(main ${src_list})

target_link_libraries(main PRIVATE lop ltensor lcuda lutils glog::glog Folly::folly Folly::folly_deps GTest::gtest)