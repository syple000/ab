add_library(cuda mem.cu info.cu tensor.cu)
target_link_libraries(cuda PUBLIC include utils cublas cudart)
target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)