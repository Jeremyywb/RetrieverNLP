
# 多卡分布式训练通信性能分析报告——以 RTX 4090 PCIe 架构为例

## 1. 报告概述

本文档对本地两卡 RTX 4090 环境下的 NCCL 通信性能进行了全面评测，评估了常见的几种 collective 操作（All-Reduce、Broadcast、All-Gather、Reduce-Scatter）在不同消息大小下的带宽表现，并基于测试结果给出优化建议。

---

## 2. 测试脚本

```bash
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

# 可选：更多集体操作
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
```

---

## 3. 系统环境

* **GPU**：2× NVIDIA GeForce RTX 4090（24 GiB，AD102）
* **互联**：PCIe-only，SHM/direct/direct；无 NVLink 支持
* **驱动/CUDA**：NVIDIA Driver 550.107.02 + CUDA 12.4
* **操作系统**：Ubuntu 22.04 LTS
* **NCCL**：v2.18.5
* **测试环境变量**：

  ```bash
  NCCL_DEBUG=INFO  
  NCCL_P2P_DISABLE=0  
  NCCL_SHM_DISABLE=0  
  NCCL_IB_DISABLE=1  
  ```

---

## 4. 测试结果汇总

| 操作类型               | 数据规模范围        | 峰值带宽        | 平均带宽        |
| ------------------ | ------------- | ----------- | ----------- |
| **All-Reduce**     | 8 B – 128 MiB | \~0.94 GB/s | \~0.45 GB/s |
| **Broadcast**      | 8 B – 128 MiB | \~1.00 GB/s | \~0.44 GB/s |
| **All-Gather**     | 8 B – 128 MiB | \~2.32 GB/s | \~0.43 GB/s |
| **Reduce-Scatter** | 8 B – 128 MiB | \~1.45 GB/s | \~0.33 GB/s |

> *注：以上结果为单次测试峰值和平均值，实际曲线可参考附录中的带宽–消息大小折线图。*

---

## 5. 深度分析

### 5.1 GPU 连接架构

* RTX 4090 不支持 NVLink，只能通过 PCIe（Gen4×16）与主机互联；
* 从 `nvidia-smi topo -m` 可见两卡跨 NUMA 节点，通信需经过 CPU-hosted SHM，进一步降低带宽。

### 5.2 通信效率特征

* **小包开销高**：<32 KB 时，各操作带宽均较低（<0.1 GB/s），由启动延迟和 kernel 调度开销主导。
* **大包性能趋稳**：≥8 MiB 时，带宽迅速上升并趋于峰值；All-Gather 尤其利用了多路径累加，达到 \~2.3 GB/s。
* **All-Reduce vs Broadcast**：二者表现相近；Reduce-Scatter 略差，因拆分／聚合跨卡请求更复杂。

### 5.3 GPU 利用率

* 测试时观察到 GPU 0 利用率 \~30%，GPU 1 \~0%，因为 NCCL benchmark 默认 rank 0 发起更多操作；
* 训练负载下应使用 `nvidia-smi dmon` 或 `nsight-sys` 做更真实的带宽与算力剖面。

---

## 6. 核心结论

> **在两卡 RTX 4090（PCIe-only）平台上，NCCL 通信最高带宽 \~2.3 GB/s，All-Reduce/ Broadcast 峰值 \~1 GB/s，平均 \~0.4–0.5 GB/s。无 NVLink 支持会成为大模型多卡训练的瓶颈。**

---

## 7. 优化与建议

1. **硬件层面**

   * 考虑更高带宽互联（NVLink/NVSwitch 或 InfiniBand）
   * 对于高通信需求，优选 A100/H100 等支持内置互联的设备

2. **软件层面**

   * 使用 ZeRO、LoRA、Gradient Accumulation 减少同步频次
   * 将梯度同步合并为大桶（`bucket_cap_mb` ≥ 256 MiB）
   * 关闭 DDP 初始化广播：`broadcast_buffers=False`、`find_unused_parameters=False`
   * 调整 NCCL 参数：

     ```bash
     export NCCL_BUFFSIZE=16777216   # 16 MiB
     export NCCL_ALGO=Tree           # 小模型场景高效
     export NCCL_SOCKET_IFNAME=eth0  # 多节点时指定
     ```

3. **运行优化**

   * 批量化通信：用 All-Gather 替代多次 All-Reduce
   * 异步通信调度：尽量 overlap 计算与通信
   * 对比 DeepSpeed ZeroStage=0，了解其跳过初始化广播的策略

