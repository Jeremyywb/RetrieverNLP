#!/bin/bash

echo "运行全面NCCL测试..."

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
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPU_COUNT-1)))

echo "=================== 系统信息 ==================="
nvidia-smi
echo "==============================================="

echo "使用CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 定义测试函数
run_test() {
  local test_name=$1
  local size_range=$2
  echo "运行 $test_name 测试..."
  echo "--------------------------------------"
  ./build/${test_name}_perf -b 8 -e $size_range -f 2 -g $GPU_COUNT
  echo "--------------------------------------"
  echo "$test_name 测试完成"
  echo
}

# 运行主要测试
run_test "all_reduce" "128M"
run_test "broadcast" "128M"
run_test "all_gather" "128M"
run_test "reduce_scatter" "128M"

# 可选：对于更详细的测试，取消下面的注释
# run_test "reduce" "128M"
# run_test "alltoall" "128M"
# run_test "scatter" "128M"
# run_test "gather" "128M"
# run_test "sendrecv" "128M"

echo "所有测试完成！"
echo
echo "================= NCCL测试总结 ================="
echo "GPU数量: $GPU_COUNT"
echo "环境变量:"
echo "  NCCL_DEBUG=$NCCL_DEBUG"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "  NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo
echo "性能提示:"
echo "1. 若要获得更好的性能，请确保GPU之间有直接的PCIe或NVLink连接"
echo "2. 对于大型集群，可能需要配置网络接口和调整NCCL参数"
echo "3. 您可以使用以下环境变量调优NCCL性能:"
echo "   - NCCL_SOCKET_IFNAME: 指定网络接口"
echo "   - NCCL_BUFFSIZE: 调整缓冲区大小"
echo "   - NCCL_RINGS: 配置NCCL通信环"
echo "   - NCCL_ALGO: 设置集体通信算法"
echo "==============================================="