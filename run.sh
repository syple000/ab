main_dir=$(pwd)
export LD_LIBRARY_PATH=$main_dir/third_party/libs:$LD_LIBRARY_PATH
cd $main_dir/build
if [[ $MEM_CHECK -eq 1 ]]; then
    echo "run in mem check mode"
    valgrind --gen-suppressions=all --suppressions=../valgrind.supp --leak-check=full ./src/main
elif [[ $CUDA_MEM_CHECK -eq 1 ]]; then 
    # 暂时不能用，原因未知
    echo "run in cuda mem check mode"
    cuda-memcheck ./src/main
else
    echo "run in normal mode"
    ./src/main
fi