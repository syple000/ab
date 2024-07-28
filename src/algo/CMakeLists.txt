add_library(algo grad_descent.cc adam.cc)
target_link_libraries(algo PUBLIC include)