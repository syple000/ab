main_dir=$(pwd)
export LD_LIBRARY_PATH=$main_dir/third_party/libs:$LD_LIBRARY_PATH
cd $main_dir/build
if [[ $MEM_CHECK -eq 1 ]]; then
    echo "run in mem check mode"
    valgrind --leak-check=full ./src/main
else
    echo "run in normal mode"
    ./src/main
fi