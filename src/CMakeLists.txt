add_subdirectory(tensor)
add_subdirectory(op)
add_subdirectory(utils)
add_subdirectory(cuda)

install(TARGETS include tensor op utils cuda 
    EXPORT ae
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(EXPORT ae
    FILE AETargets.cmake
    NAMESPACE AE::
    DESTINATION lib/cmake/AE)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/AEConfig.cmake.in ${PROJECT_BINARY_DIR}/AEConfig.cmake
  INSTALL_DESTINATION lib/cmake/AE)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/AEConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES "${PROJECT_BINARY_DIR}/AEConfig.cmake"
              "${PROJECT_BINARY_DIR}/AEConfigVersion.cmake"
        DESTINATION lib/cmake/AE)

add_executable(main main.cc)
target_link_libraries(main include tensor op cuda utils GTest::gtest glog::glog)