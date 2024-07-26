add_library(include INTERFACE)
# include目录在项目本地构建 & 安装后引用目录有差异
target_include_directories(include 
    INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(include INTERFACE GTest::gtest glog::glog pybind11::headers Python::Python fmt::fmt)
set_target_properties(include PROPERTIES 
    INTERPROCEDURAL_OPTIMIZATION ON
    CXX_VISIBILITY_PRESET ON
    VISIBILITY_INLINES_HIDDEN ON)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/auto_engine DESTINATION include)