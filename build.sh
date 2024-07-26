set -e # 任何错误都及时退出，防止误删文件

# 项目目录
main_dir=$(pwd)
echo "project dir: " $main_dir

# 主项目编译，并拷贝编译用json
mkdir -p $main_dir/build
cd $main_dir/build
cmake .. && make

mv compile_commands.json .. # clangd

mv lib/ae\.* ../py # py
cd $main_dir/py
export PYTHONPATH=$main_dir/py:$PYTHONPATH
pybind11-stubgen ae
mv stubs/* .
rm -r stubs && rm -f ae.INFO ae.WARNING ae.ERROR info.log* warning.log* error.log*

cd $main_dir