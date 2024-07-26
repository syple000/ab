add_subdirectory(cuda)
add_subdirectory(tensor)
add_subdirectory(op)
add_subdirectory(utils)
add_subdirectory(algo)
add_subdirectory(py)

install(TARGETS include tensor op utils cuda 
    EXPORT auto_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(EXPORT auto_engine
    FILE AETargets.cmake
    NAMESPACE auto_engine::
    DESTINATION share/auto_engine)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/auto_engine-config.cmake.in 
  ${PROJECT_BINARY_DIR}/auto_engine-config.cmake
  INSTALL_DESTINATION share/auto_engine)

install(FILES 
    "${PROJECT_BINARY_DIR}/auto_engine-config.cmake"
    DESTINATION
    share/auto_engine)

add_executable(main main.cc)
target_link_libraries(main include tensor op utils cuda algo)