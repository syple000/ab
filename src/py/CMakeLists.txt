pybind11_add_module(ae tensor_f64.cc)
target_link_libraries(ae PRIVATE include tensor op utils cuda algo)
