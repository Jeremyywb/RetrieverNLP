#!/bin/bash

# 创建一个函数来创建符号链接和所需目录
create_link() {
    local source=$1
    local target=$2
    local target_dir=$(dirname "$target")
    
    if [ ! -d "$target_dir" ]; then
        echo "创建目录: $target_dir"
        sudo mkdir -p "$target_dir"
    fi
    
    echo "创建符号链接: $source -> $target"
    sudo ln -sf "$source" "$target"
}

echo "开始修复OpenMPI头文件..."

# 查找所有需要的头文件
MPI_INCLUDE_DIR="/usr/lib/x86_64-linux-gnu/openmpi/include"
echo "OpenMPI包含文件目录: $MPI_INCLUDE_DIR"

# 1. 创建基本的MPI头文件链接
create_link "$MPI_INCLUDE_DIR/mpi.h" "/usr/include/mpi.h"
create_link "$MPI_INCLUDE_DIR/mpi_portable_platform.h" "/usr/include/mpi_portable_platform.h"
create_link "$MPI_INCLUDE_DIR/mpi-ext.h" "/usr/include/mpi-ext.h"

# 2. 创建OpenMPI C++头文件目录结构
OMPI_CXX_DIR="/usr/include/openmpi/ompi/mpi/cxx"

# 链接所有 cxx 目录下的头文件
for file in $(find $MPI_INCLUDE_DIR/openmpi/ompi/mpi/cxx -name "*.h" 2>/dev/null); do
    basename=$(basename "$file")
    create_link "$file" "$OMPI_CXX_DIR/$basename"
done

echo "测试NVCC能否处理MPI头文件..."
echo '#include <mpi.h>' | nvcc -E -x cu - -I/usr/include -o /dev/null

if [ $? -eq 0 ]; then
    echo "成功! MPI头文件已正确配置。"
else
    echo "仍然存在问题。检查错误消息并尝试手动修复。"
fi