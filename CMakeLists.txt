cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
project(auto_engine VERSION 0.0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
add_compile_options(-Wno-unused-result)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 

# clangd对cuda输出的commands.json支持不好，想要正确识别，请删除：/usr/share/cmake-3.28/Modules/Compiler/NVIDIA.cmake
# 中RESPONSE_FILE & options-file相关内容
enable_language(CUDA)

find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)

add_subdirectory(include)
add_subdirectory(src)