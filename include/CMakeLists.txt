add_library(include INTERFACE)
# include目录在项目本地构建 & 安装后引用目录有差异
target_include_directories(include 
    INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/auto_engine>
    $<INSTALL_INTERFACE:include/auto_engine>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/auto_engine DESTINATION include)