set -e # 任何错误都及时退出，防止误删文件

# 项目目录
main_dir=$(pwd)
echo "project dir: " $main_dir

third_src_dir=$main_dir/third_party/srcs
third_include_dir=$main_dir/third_party/includes
third_lib_dir=$main_dir/third_party/libs
mkdir -p $third_lib_dir
mkdir -p $third_include_dir
# glog项目
if [ ! -e $third_lib_dir/libglog.so ]; then
    mkdir -p $third_src_dir/glog/build
    mkdir -p $third_include_dir/glog

    cd $third_src_dir/glog/build
    cmake .. && make

    target_so=$(readlink -f libglog.so)
    if [ -e $target_so ]; then
        cp $target_so $third_lib_dir/libglog.so
        ln -s $third_lib_dir/libglog.so $third_lib_dir/libglog.so.2
    else
        echo "target glog so not found"
        exit -1
    fi

    cp glog/export.h $third_include_dir/glog
    cp $third_src_dir/glog/src/glog/* $third_include_dir/glog

    cd -
fi

# gtest项目
if [ ! -e $third_lib_dir/libgtest.a ]; then
    mkdir -p $third_src_dir/gtest/build
    mkdir -p $third_include_dir/gtest

    cd $third_src_dir/gtest/build
    cmake .. && make

    cp lib/libgtest.a $third_lib_dir
    cp -r $third_src_dir/gtest/googletest/include/gtest/* $third_include_dir/gtest 
fi

# 主项目编译，并拷贝编译用json
mkdir -p $main_dir/build
cd $main_dir/build
cmake .. && make
mv compile_commands.json .. 

cd -