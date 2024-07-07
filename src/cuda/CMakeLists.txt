add_library(lcuda mem.cu)
target_include_directories(lcuda PUBLIC ${PROJECT_SOURCE_DIR}/include/auto_engine)