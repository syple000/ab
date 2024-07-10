add_library(cuda mem.cu matrix_f64.cu)
target_link_libraries(cuda PUBLIC include utils cublas cudart)