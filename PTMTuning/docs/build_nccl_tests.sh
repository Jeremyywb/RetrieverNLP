#!/bin/bash

echo "编译NCCL测试..."

# 检查当前目录是否已经有nccl-tests
if [ -d "nccl-tests" ]; then
  echo "发现现有的nccl-tests目录，直接使用"
  cd nccl-tests
else
  echo "克隆NCCL测试代码库..."
  git clone https://github.com/NVIDIA/nccl-tests.git
  cd nccl-tests
fi

# 检查CUDA路径
if [ -z "$CUDA_HOME" ]; then
  # 尝试设置CUDA_HOME的默认路径
  if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
  else
    # 寻找cuda路径
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    if [ -n "$CUDA_PATH" ]; then
      export CUDA_HOME=$CUDA_PATH
    else
      echo "错误: 找不到CUDA路径。请设置CUDA_HOME环境变量。"
      exit 1
    fi
  fi
fi

echo "使用CUDA_HOME: $CUDA_HOME"

# 检查MPI路径
MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi"
echo "使用MPI_HOME: $MPI_HOME"

# 检查NCCL路径
if [ -z "$NCCL_HOME" ]; then
  # 尝试找到nccl.h
  NCCL_H_PATH=$(find /usr -name nccl.h 2>/dev/null | head -1)
  if [ -n "$NCCL_H_PATH" ]; then
    export NCCL_HOME=$(dirname $(dirname $NCCL_H_PATH))
    echo "自动设置NCCL_HOME: $NCCL_HOME"
  else
    echo "警告: 找不到NCCL路径。如果编译失败，请安装NCCL或设置NCCL_HOME环境变量。"
  fi
fi

# 显示环境变量
echo "环境变量:"
echo "CUDA_HOME=$CUDA_HOME"
echo "MPI_HOME=$MPI_HOME"
echo "NCCL_HOME=$NCCL_HOME"

# 开始编译
echo "开始编译NCCL测试..."
make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME -j $(nproc)

# 检查编译结果
if [ $? -eq 0 ]; then
  echo "编译成功!"
  echo "可执行文件位于: $(pwd)/build"
  ls -la build/
else
  echo "编译失败。请检查错误信息。"
fi