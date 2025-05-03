
# 大模型训练中的显存优化技术：ZeRO技术详解与实践

## 零、All-Gather，Reduce-Scatter，All-Reduce

---

### 为何要关注通信原语？

在分布式训练中，**通信**往往比计算更容易成为瓶颈。理解和合理运用通信原语，不仅能保证模型正确同步，还能最大化带宽利用、最小化显存占用，从而提升整体吞吐率。

---

### 0.1 数据汇聚：让每台设备看到“全景画面”

#### 场景  
- 统计全局损失或指标  
- 需要每个节点都持有相同的参数快照  

#### 核心操作  
- **“收集–拼接”**：每台设备把自己的那份数据丢到网络上，循环传递 N−1 轮，就能在本地拼出大家的全部数据。

```text
节点 i 发送自己第 i 段 → 接收上家分块  
重复 N−1 轮 → 拥有所有 N 段  
```

> 这种模式就是 **All-Gather**：保证所有节点拥有同一份“全景画面”。

---

### 0.2 数据裁剪：只给你该看的一部分

#### 场景  
- ZeRO 分片梯度或参数  
- 只想把“自己负责”的那段数据存起来  

#### 核心操作  
1. **聚合（Reduce）**：在每个位置把 N 台机器的数据求和或求平均。  
2. **分发（Scatter）**：把结果再切成 N 段——每台只留自己那一段。

```text
所有节点第 j 段数据 → 求和 → 分发给节点 j  
```

> 这就是 **Reduce-Scatter**：既做了规约，又只保留自己的一小块，大幅削减显存占用。

---

###  0.3 数据合一：一把梳理再广播

#### 场景  
- 需要把全局梯度一起算完再更新所有参数  
- 既要规约，又要让每台都用上新结果  

#### 分两步走  
1. **先裁剪**（Reduce-Scatter）  
2. **再拼接**（All-Gather）  

合在一起就形成了**All-Reduce**。主流实现多用环形（Ring All-Reduce），在 2(N−1) 步里完成。

```text
步骤 1–(N−1)：分片规约 → 每台有局部和  
步骤 N–2N−2：逐段广播 → 每台得全局和  
```

> 高效、可扩展，DeepSpeed、PyTorch DDP、Megatron-LM 都在用。

---

### 0.4 通信管道：流水线 vs 整体阻塞

- **一次性通信**：等所有梯度算完，再集中发送。简单但会有通信峰值。  
- **流水线通信**：Gradients Ready → 立即分桶（bucket）通信，与后续层反向并行。DeepSpeed 的 **通信重叠**（Overlap Comm）恰是如此：  
  1. 张量梯度分成若干小桶  
  2. 每桶一 ready 就发起 Reduce-Scatter  
  3. 后向还在进行，其它桶继续触发

> 把那一次 All-Reduce 拆成碎片，最大化带宽利用，隐藏通信延迟。

---

### 0.5 分布式优化：把累积与通信合二为一

1. **智能累积**（Smart Accumulation）：Micro-batch 内部梯度先本地累加，**不跨卡**；  
2. **最终平均**：累加完后，再对整个梯度做一次全局平均（All-Reduce），然后更新；  
3. **和重叠通信**：把这一次平均拆桶并行发，和最后几层的反向重叠跑。

```text
for each micro_batch:
  loss.backward()        # 仅本地累加
# 结束后
gradients /= k           # 平均
trigger bucket‐by‐bucket Reduce-Scatter (overlap)  
optimizer.step()
```



## 一、大模型训练中的显存消耗分析

### 1.1 显存消耗构成

在深入分析ZeRO技术前，我们首先需要清晰地了解大模型训练过程中GPU显存的使用情况。训练过程中的显存消耗可以分为两大类：

#### 1.1.1 模型状态（Model States）

这部分是与模型结构直接相关、训练过程中必须存储的内容：

- **模型参数（Parameters）**：网络的权重矩阵，通常记为W，是模型的核心组成部分
- **参数梯度（Gradients）**：反向传播过程中计算得到的梯度，用于参数更新
- **优化器状态（Optimizer States）**：如Adam优化器中的一阶矩估计（动量）和二阶矩估计（方差），用于调整参数更新步长和方向

#### 1.1.2 残余状态（Residual States）

这部分是训练过程中产生的、并非绝对必要但影响效率的内容：

- **激活值（Activations）**：前向传播中每一层的输出。虽然可以通过重计算获得，但存储它们可以加速反向传播
- **临时缓冲区（Temporary Buffers）**：如通信过程中的发送/接收缓冲区、计算中间结果的临时存储等
- **内存碎片（Memory Fragments）**：由于显存分配和释放过程中产生的不连续空间，虽然显示为已用但实际无法利用

### 1.2 混合精度训练与显存占用

为了更高效地利用计算资源，现代大模型训练普遍采用混合精度训练策略。基于FP16/BF16与FP32混合的训练流程如下：

1. **维护FP32主副本**：存储一份FP32精度的模型参数和优化器状态（动量和方差）
2. **创建FP16工作副本**：训练前将FP32参数转换为FP16，用于实际计算
3. **半精度计算**：使用FP16参数进行前向和反向传播，生成FP16梯度
4. **梯度处理与参数更新**：将FP16梯度转换回FP32精度，然后更新FP32主参数
5. **循环迭代**：每次迭代重复上述过程

假设模型参数量为$\Phi$，在混合精度训练下，主要显存占用为：

- FP32参数：$4\Phi$ 字节
- FP16参数副本：$2\Phi$ 字节
- Adam优化器状态（FP32）：$8\Phi$ 字节（每个参数对应两个状态变量）
- FP16梯度：$2\Phi$ 字节

这意味着仅模型状态就需要 $16\Phi$ 字节的显存，对于一个拥有1750亿参数的GPT-3级别模型，需要约2.8TB的显存！而当前最高端的NVIDIA A100 GPU仅有80GB显存，显然存在巨大缺口。

## 二、ZeRO技术原理：消除冗余，按需计算

ZeRO（Zero Redundancy Optimizer）是由微软DeepSpeed团队提出的一系列显存优化技术，其核心思想是将数据并行训练中的冗余存储消除，通过在GPU间分片存储模型状态，实现显存需求的线性缩减。随着分片范围的扩大，ZeRO技术分为三个渐进式的阶段：

### 2.1 数据并行中的冗余问题

在传统数据并行训练中，每个GPU都保存完整的模型副本（包括参数、梯度和优化器状态），仅对数据进行分片处理。这种方式虽然实现简单，但在扩展到超大规模模型时会导致严重的显存冗余。

### 2.2 ZeRO-1：优化器状态分片

**优化策略**：仅对优化器状态（动量和方差）进行分片存储

#### 实现机制：

1. **状态分片**：将优化器状态按参数索引均匀分配到N个GPU上，每个GPU仅存储约$\frac{1}{N}$的优化器状态
2. **前向传播**：每个GPU持有完整的FP32参数和FP16参数副本，独立执行前向计算
3. **反向传播**：计算本地梯度，然后通过AllReduce操作聚合所有GPU的梯度
4. **参数更新**：每个GPU仅更新自己负责的那部分参数对应的优化器状态，然后更新这部分参数
5. **参数同步**：通过AllGather操作重建完整参数

#### 显存占用分析：
- FP32参数：$4\Phi$ 字节
- FP16参数副本：$2\Phi$ 字节
- 分片优化器状态：$\frac{8\Phi}{N}$ 字节
- 梯度：$2\Phi$ 字节

ZeRO-1可将优化器状态的显存占用降低为原来的$\frac{1}{N}$，总显存需求约为$(8+\frac{8}{N})\Phi$字节。

### 2.3 ZeRO-2：梯度分片

**优化策略**：在ZeRO-1基础上进一步对梯度进行分片存储

#### 实现机制：

1. **前向传播**：
   - 每个GPU持有完整的FP32主参数
   - 训练前将FP32参数复制为FP16参数副本用于计算
   - 使用本地数据批次执行前向计算，生成激活值

2. **反向传播**：
   - 每个GPU计算本地梯度（FP16格式）
   - 通过Reduce-Scatter操作进行全局梯度聚合与分片
   - **关键细节**：Reduce-Scatter完成后，每个GPU仅保留自己负责的参数部分对应的梯度分片，其余梯度分片被丢弃
   - 这种"计算全部、保留部分"的策略是ZeRO-2显存优化的核心

3. **参数更新**：
   - 每个GPU仅使用自己持有的梯度分片更新对应的FP32参数部分
   - 同时，仅更新这部分参数对应的优化器状态（动量和方差）
   - 参数更新公式：$W_i^{(t+1)} = W_i^{(t)} - \eta \cdot \text{Adam}(G_i^{(t)}, M_i^{(t)}, V_i^{(t)})$
   - 其中$i$表示当前GPU负责的参数分片索引

4. **参数同步**：
   - 所有GPU通过AllGather操作重建完整的FP32参数
   - 再次复制为FP16参数副本，用于下一轮迭代的计算

#### 显存占用分析：
- FP32参数：$4\Phi$ 字节
- FP16参数副本：$2\Phi$ 字节
- 分片优化器状态：$\frac{8\Phi}{N}$ 字节
- 分片梯度：$\frac{2\Phi}{N}$ 字节

ZeRO-2将优化器状态和梯度的显存占用均降低为原来的$\frac{1}{N}$，总显存需求约为$(6+\frac{10}{N})\Phi$字节。典型情况下，对于8GPU系统，理论上可以减少约56%的显存占用。

### 2.4 ZeRO-3：参数分片

**优化策略**：在ZeRO-2基础上，连模型参数也进行分片存储，实现完全分片

#### 实现机制：

1. **参数分片**：
   - 模型参数（FP32和FP16）按维度均匀分配到N个GPU
   - 每个GPU仅存储约$\frac{1}{N}$的模型参数
   - 这是ZeRO-3与前两个阶段的本质区别：其他GPU不再保留完整参数副本

2. **前向传播**：
   - 执行每一层计算前，通过AllGather操作临时重建该层所需的完整参数
   - **关键细节**：重建是按层或按模块动态进行的，而非一次性重建整个模型
   - 完成当前层计算后立即释放重建的参数，以最小化显存占用
   - 这种"按需重建、用完即弃"的策略使ZeRO-3能处理超大模型

3. **反向传播**：
   - 同样按层序执行，需要时再次通过AllGather临时重建参数
   - 计算该层梯度
   - 通过Reduce-Scatter操作聚合全局梯度并分片到对应GPU
   - **与ZeRO-2的区别**：ZeRO-3在反向传播中也需要频繁重建参数

4. **参数更新**：
   - 每个GPU仅更新自己负责的参数分片和对应的优化器状态
   - 无需额外的参数同步步骤，因为下一轮迭代时会再次按需重建
   - 这种方式大幅减少了常驻显存需求，但增加了通信开销

#### 显存占用分析：
- 分片FP32参数：$\frac{4\Phi}{N}$ 字节
- 临时FP16参数（按需重建）：约$\frac{2\Phi}{L}$ 字节（L为模型层数）
- 分片优化器状态：$\frac{8\Phi}{N}$ 字节
- 分片梯度：$\frac{2\Phi}{N}$ 字节

ZeRO-3将所有模型状态的显存占用均降低为原来的$\frac{1}{N}$，总显存需求约为$\frac{14\Phi}{N}+\frac{2\Phi}{L}$字节，实现了近乎完美的显存线性缩放。

## 三、ZeRO技术实现细节与优化

### 3.1 通信优化与计算重叠

ZeRO虽然大幅降低了显存需求，但也引入了额外的通信开销，特别是在ZeRO-3中。DeepSpeed通过一系列精细优化来减轻这些通信开销的影响：

1. **分桶通信（Bucketing）**：
   - 将参数按大小分成多个桶（buckets）
   - 通信以桶为单位进行，而非等待所有参数准备就绪
   - 通过`reduce_bucket_size`和`allgather_bucket_size`参数控制桶大小
   - 适当的桶大小可以减少通信启动延迟，同时避免过大的通信包

2. **通信与计算重叠**：
   - 利用CUDA流（streams）和NCCL库的异步特性
   - 在等待一个参数桶通信完成的同时执行其他计算
   - 启用`overlap_comm`选项使通信操作与计算并行执行
   - 这种重叠可以隐藏大部分通信延迟

3. **预取机制（Prefetch）**：
   - 提前预测并启动下一层参数的AllGather操作
   - 与当前层的计算并行执行
   - 通过`stage3_prefetch_bucket_size`控制预取粒度
   - 在前向和反向传播中均可应用这一策略

4. **通信优化器**：
   - 自适应选择最佳通信原语（如NCCL或Gloo）
   - 动态调整通信顺序以最小化等待时间
   - 利用拓扑感知算法优化多节点环境中的通信路径

这些精细的通信优化使ZeRO的额外通信开销在实际训练中大幅降低，在良好的网络环境下（如InfiniBand或NVLink互联）几乎可以被完全隐藏，从而保持训练吞吐量不受太大影响。

### 3.2 激活值优化

除了模型状态，激活值也是大模型训练中的显存大户。DeepSpeed结合ZeRO提供了多种激活值优化策略：

1. **激活检查点（Activation Checkpointing）**：只保存关键层的激活值，其他层在反向传播时重计算
2. **激活分片（Activation Partitioning）**：在ZeRO-3中，激活值也可以按数据维度分片存储
3. **CPU卸载（CPU Offloading）**：将不立即使用的激活值临时存储到CPU内存

### 3.3 CPU与NVMe卸载

为进一步扩展可训练的模型规模，ZeRO-Offload和ZeRO-Infinity扩展了ZeRO的分片思想：

1. **ZeRO-Offload**：将优化器状态和部分梯度卸载到CPU内存
2. **ZeRO-Infinity**：将模型状态扩展到NVMe存储，理论上可训练无限大的模型

这些技术结合使用时，即使在单台多GPU服务器上也能训练数千亿参数的模型。

### 3.4 各阶段特点与选择指南

| 特性 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| **分片对象** | 优化器状态 | 优化器状态 + 梯度 | 参数 + 梯度 + 优化器 |
| **显存节省** | ~30% | ~55% | ~70% |
| **通信开销** | 低 | 中 | 高 |
| **实现复杂度** | 低 | 中 | 高 |
| **适用模型规模** | 10B-40B | 40B-175B | 175B+ |

选择合适的ZeRO阶段应考虑：
- 模型规模
- 可用GPU数量和类型
- 网络通信带宽
- 训练吞吐量要求

对中小规模模型（<40B参数），ZeRO-1通常足够；对超大规模模型，ZeRO-3是必要选择。

### 3.5 参数拷贝的显存管理

- **动态释放策略**：  
  FP16计算副本在前向传播后立即释放，避免长期占用显存。  
  ```python
  del fp16_params  # 显存释放 2Φ
  ```

- **内存预分配**：  
  预分配FP16参数缓冲区，减少动态分配的开销。  
  ```python
  buffer = torch.empty_like(fp32_master_params, dtype=torch.float16)
  ```

### 3.6 时间换空间策略**
1. **梯度累积（Gradient Accumulation）**：  
   - 多批次梯度累加后更新，减少优化器状态访问频率。  
   - **显存节省**：梯度显存保持 \(4\Phi\)，但批次减至 \(B/K\)。  

2. **分阶段加载数据**：  
   - 按需加载数据块，避免全量数据显存占用。  
   - **显存节省**：数据显存从 \(B \times D\) 降至 \(D\)（\(D\)=单样本维度）。




## 四、ZeRO在实践中的应用与配置



### 4.2 DeepSpeed配置示例

使用PyTorch和DeepSpeed实现ZeRO优化的配置示例（ZeRO-3带CPU卸载）：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_fp16_weights_on_model_save": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

### 4.3 关键参数解析与优化建议

#### 核心参数详解

1. **stage**：选择ZeRO阶段（1、2、3）
   - 小型模型（<10B参数）：推荐使用stage 1或2
   - 大型模型（>100B参数）：推荐使用stage 3
   - 在通信受限环境中，较低阶段可能更优

2. **offload配置**：
   - **offload_optimizer**：将优化器状态卸载到CPU，可节省大量显存
   - **offload_param**：将参数卸载到CPU，适用于极限场景
   - **pin_memory**：使用固定内存提升CPU-GPU传输速度
   - 卸载会引入CPU-GPU数据传输开销，但对于显存受限场景非常有用

3. **通信参数**：
   - **overlap_comm**：启用通信与计算重叠，几乎必选
   - **reduce_bucket_size**：梯度聚合桶大小，建议值约为0.5GB-1GB
   - **stage3_prefetch_bucket_size**：ZeRO-3参数预取桶大小
   - 较大的桶大小减少通信启动次数，但增加延迟；较小的桶大小利于重叠，但增加启动开销

4. **内存优化参数**：
   - **contiguous_gradients**：合并碎片化梯度缓冲区，减少内存碎片
   - **stage3_param_persistence_threshold**：小于此阈值的参数常驻GPU不分片
   - **stage3_max_live_parameters**：控制重建参数的最大内存占用
   - **stage3_max_reuse_distance**：控制参数重用距离，影响缓存策略

#### 实际场景优化建议

1. **显存极度受限场景**（如消费级GPU训练大模型）：
   ```json
   {
     "zero_optimization": {
       "stage": 3,
       "offload_optimizer": {"device": "cpu"},
       "offload_param": {"device": "cpu"},
       "overlap_comm": true,
       "contiguous_gradients": true,
       "reduce_bucket_size": 2e8
     }
   }
   ```

2. **高性能训练场景**（如A100集群训练中等规模模型）：
   ```json
   {
     "zero_optimization": {
       "stage": 2,
       "offload_optimizer": {"device": "none"},
       "overlap_comm": true,
       "contiguous_gradients": true,
       "reduce_bucket_size": 7e8
     }
   }
   ```

3. **混合并行训练超大模型**（如训练万亿参数模型）：
   ```json
   {
     "zero_optimization": {
       "stage": 3,
       "offload_optimizer": {"device": "cpu"},
       "stage3_param_persistence_threshold": 1e6,
       "overlap_comm": true
     },
     "pipeline": {
       "enabled": true,
       "stages": 4
     }
   }
   ```

### 4.3 与其他并行技术的结合

ZeRO可以与其他并行训练技术结合使用，形成更强大的分布式训练方案：

1. **ZeRO + 模型并行**：ZeRO处理数据并行部分的显存优化，模型并行分割超大层
2. **ZeRO + 流水线并行**：将模型按层分割到不同GPU，配合ZeRO处理每个阶段的参数
3. **ZeRO + 张量并行**：对单个大型算子内部进行并行计算，ZeRO处理剩余部分

## 五、ZeRO技术实践中的常见问题与解决方案

### 5.1 内存碎片问题

在实际训练中，特别是使用ZeRO-3时，显存碎片化是一个常见问题：

1. **问题表现**：
   - 虽然总显存充足，但申请大块连续内存时失败
   - 训练中期或后期突然出现OOM错误
   - nvidia-smi显示的使用率低于预期但仍OOM

2. **解决方案**：
   - 启用`contiguous_gradients`选项合并梯度缓冲区
   - 调整`reduce_scatter_bucket_size`控制通信颗粒度
   - 设置`round_robin_gradients`让梯度分配更均匀
   - 在模型定义中避免动态创建过多中间张量

### 5.2 通信瓶颈问题

多节点训练中，通信往往成为ZeRO扩展性的瓶颈：

1. **问题表现**：
   - GPU利用率低，大部分时间在等待通信
   - 增加GPU数量后，训练速度提升不明显甚至下降

2. **解决方案**：
   - 优先使用高带宽互联（如InfiniBand或NVLink）
   - 调整通信桶大小，在ZeRO-3中设置合理的预取参数
   - 针对特定拓扑结构优化通信算法
   - 考虑结合张量并行减少通信量

### 5.3 参数更新的数值稳定性

ZeRO的分片策略可能影响训练的数值稳定性：

1. **问题表现**：
   - 分布式训练结果与单机训练不一致
   - 不同GPU数量下结果不可复现
   - 训练损失不稳定或梯度爆炸

2. **解决方案**：
   - 使用确定性算法（`--deterministic`）
   - 调整梯度裁剪阈值以稳定训练
   - 在ZeRO-3中使用`fp32_reduce_scatter`选项提高精度
   - 考虑使用BF16替代FP16以获得更好的数值稳定性

