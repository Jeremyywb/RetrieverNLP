#!/bin/bash

echo "运行修复的NCCL测试..."

# 进入nccl-tests目录
cd nccl-tests

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 1 ]; then
  echo "错误: 没有检测到GPU。NCCL测试需要至少一个GPU。"
  exit 1
fi

# 设置关键环境变量
export NCCL_DEBUG=INFO
# 允许单机多卡通信
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# 禁用IB通信（Docker容器中可能没有配置好）
export NCCL_IB_DISABLE=1
# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0$([ $GPU_COUNT -gt 1 ] && echo ",1")

echo "=================== 系统信息 ==================="
nvidia-smi
echo "==============================================="

echo "使用CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 尝试单进程模式
echo "尝试单进程模式运行all_reduce测试..."
./build/all_reduce_perf -b 8 -e 8M -f 2 -g $GPU_COUNT

if [ $? -eq 0 ]; then
  echo "单进程模式成功！"
else
  echo "单进程模式失败，尝试MPI模式..."
  
  # 尝试使用MPI，但每个进程只使用一个GPU
  for i in $(seq 0 $(($GPU_COUNT-1))); do
    echo "测试GPU $i..."
    CUDA_VISIBLE_DEVICES=$i ./build/all_reduce_perf -b 8 -e 8M -f 2 -g 1
  done
  
  echo "如果需要使用MPI进行多GPU测试，请尝试以下命令:"
  echo "mpirun -n $GPU_COUNT --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -b 8 -e 8M -f 2 -g 1"
fi

echo "测试完成。如果遇到问题，您可能需要检查NCCL配置或GPU可见性设置。"