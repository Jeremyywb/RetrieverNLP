#!/bin/bash

echo "运行NCCL测试..."

# 进入nccl-tests目录
cd nccl-tests

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 1 ]; then
  echo "错误: 没有检测到GPU。NCCL测试需要至少一个GPU。"
  exit 1
fi

# 设置运行参数
if [ $GPU_COUNT -eq 1 ]; then
  # 单GPU模式，使用2个进程在同一个GPU上运行
  NGPUS=1
  PROC_PER_NODE=2
  echo "单GPU模式: 将在同一GPU上运行2个进程"
else
  # 多GPU模式，每个GPU一个进程
  NGPUS=$GPU_COUNT
  PROC_PER_NODE=$GPU_COUNT
  echo "多GPU模式: 每个GPU运行1个进程"
fi

# 确保build目录存在
if [ ! -d "build" ]; then
  echo "错误: 找不到build目录，请确保NCCL测试已经编译成功。"
  exit 1
fi

# 设置NCCL环境变量
export NCCL_DEBUG=INFO

echo "=================== 系统信息 ==================="
nvidia-smi
echo "==============================================="

echo "运行简单的all_reduce测试..."
mpirun -n $PROC_PER_NODE --allow-run-as-root ./build/all_reduce_perf -b 8 -e 128M -f 2 -g $NGPUS

echo "运行简单的broadcast测试..."
mpirun -n $PROC_PER_NODE --allow-run-as-root ./build/broadcast_perf -b 8 -e 128M -f 2 -g $NGPUS

echo "测试完成。"
echo "如果需要运行其他测试，您可以使用以下命令："
echo "mpirun -n [进程数] --allow-run-as-root ./build/[测试名称]_perf [参数]"
echo "可用的测试名称: all_reduce, broadcast, all_gather, reduce_scatter, reduce, alltoall, scatter, gather, sendrecv, hypercube"