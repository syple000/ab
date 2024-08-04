add_library(algo opt_grad_descent.cc opt_adam.cc loss.cc)
target_link_libraries(algo PUBLIC include)