好的！下面是一个完整的、**可直接在 Linux 命令行中执行的脚本流程**，用于：

1. 安装依赖（CUDA、NCCL、OpenMPI）
2. 下载 `nccl-tests` 源码
3. 编译 `nccl-tests`
4. 运行常见 NCCL 通信测试（如 all_reduce、all_gather 等）

---

## ✅ **前提条件**
- 你使用的是 **Ubuntu/Debian** 或兼容的 Linux 发行版。
- 已安装 NVIDIA 驱动。
- 具有 root 权限或 sudo 权限。

---

## 🔧 第一步：安装依赖项

```bash
# 更新系统包列表
sudo apt update

# 安装基础构建工具
sudo apt install -y build-essential git wget

# 安装 CUDA Toolkit（以 12.1 为例，根据你的 GPU 选择合适版本）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# 安装 NCCL（推荐通过 CUDA 包含的版本即可）
# 如果需要最新版本，可以单独从官网下载并安装 .deb 包

# 安装 OpenMPI（用于多进程通信）
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```

---

## 📦 第二步：下载 nccl-tests 源码

```bash
# 克隆仓库
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
```

---

## ⚙️ 第三步：编译 nccl-tests

claude.ai获取的自动化编译脚本成功测试。

---

参考如下沟通内容：

AI:
看起来您在尝试配置NCCL测试环境时遇到了OpenMPI头文件的问题。您已经安装了OpenMPI，但在使用nvcc编译包含MPI头文件的代码时遇到了缺失头文件的问题。我将一步步指导您解决这个问题。

首先，让我们尝试创建一个脚本来系统地解决缺失的头文件问题：

请执行以下命令，赋予脚本执行权限并运行它：

```bash
chmod +x fix_mpi_headers.sh
./fix_mpi_headers.sh
```

执行后，请将输出结果反馈给我，我会根据结果进行下一步指导。

USER:
测试NVCC能否处理MPI头文件...
成功! MPI头文件已正确配置。

AI:

太好了！OpenMPI的头文件问题已经成功解决。现在NVCC可以正确处理MPI头文件了。

让我们继续配置您的NCCL测试环境。下一步是确认我们可以编译NCCL测试。我将创建一个脚本来克隆和编译NCCL测试：

请执行以下命令，赋予脚本执行权限并运行它：

```bash
chmod +x build_nccl_tests.sh
./build_nccl_tests.sh
```

这个脚本会：
1. 检查是否已有nccl-tests目录，否则从GitHub克隆代码
2. 自动检测CUDA、MPI和NCCL的路径
3. 使用这些路径编译NCCL测试

执行后，请将输出结果反馈给我，我会根据结果进行下一步指导。

USER:
(myproject_env) (base) root@0fed198eb8ba:/home# chmod +x build_nccl_tests.sh
./build_nccl_tests.sh
编译NCCL测试...
发现现有的nccl-tests目录，直接使用
使用CUDA_HOME: /usr/local/cuda
使用MPI_HOME: /usr/lib/x86_64-linux-gnu/openmpi
自动设置NCCL_HOME: /usr
环境变量:
CUDA_HOME=/usr/local/cuda
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
NCCL_HOME=/usr
开始编译NCCL测试...
make -C src build BUILDDIR=/home/nccl-tests/build
make[1]: 进入目录“/home/nccl-tests/src”
Compiling /home/nccl-tests/build/verifiable/verifiable.o
Compiling  common.cu                           > /home/nccl-tests/build/common.o
Compiling  all_reduce.cu                       > /home/nccl-tests/build/all_reduce.o
Compiling  all_gather.cu                       > /home/nccl-tests/build/all_gather.o
Compiling  broadcast.cu                        > /home/nccl-tests/build/broadcast.o
Compiling  reduce_scatter.cu                   > /home/nccl-tests/build/reduce_scatter.o
Compiling  reduce.cu                           > /home/nccl-tests/build/reduce.o
Compiling  alltoall.cu                         > /home/nccl-tests/build/alltoall.o
Compiling  scatter.cu                          > /home/nccl-tests/build/scatter.o
Compiling  gather.cu                           > /home/nccl-tests/build/gather.o
Compiling  sendrecv.cu                         > /home/nccl-tests/build/sendrecv.o
Compiling  hypercube.cu                        > /home/nccl-tests/build/hypercube.o
Linking  /home/nccl-tests/build/all_reduce.o > /home/nccl-tests/build/all_reduce_perf
Linking  /home/nccl-tests/build/broadcast.o  > /home/nccl-tests/build/broadcast_perf
Linking  /home/nccl-tests/build/all_gather.o > /home/nccl-tests/build/all_gather_perf
Linking  /home/nccl-tests/build/reduce_scatter.o > /home/nccl-tests/build/reduce_scatter_perf
Linking  /home/nccl-tests/build/reduce.o     > /home/nccl-tests/build/reduce_perf
Linking  /home/nccl-tests/build/alltoall.o   > /home/nccl-tests/build/alltoall_perf
Linking  /home/nccl-tests/build/scatter.o    > /home/nccl-tests/build/scatter_perf
Linking  /home/nccl-tests/build/gather.o     > /home/nccl-tests/build/gather_perf
Linking  /home/nccl-tests/build/sendrecv.o   > /home/nccl-tests/build/sendrecv_perf
Linking  /home/nccl-tests/build/hypercube.o  > /home/nccl-tests/build/hypercube_perf
make[1]: 离开目录“/home/nccl-tests/src”
编译成功!
可执行文件位于: /home/nccl-tests/build
总计 31
drwxr-xr-x 3 root root       14  5月  2 20:24 .
drwxr-xr-x 7 root root       11  5月  2 19:59 ..
-rwxr-xr-x 1 root root 18233480  5月  2 20:24 all_gather_perf
-rwxr-xr-x 1 root root 18233760  5月  2 20:24 all_reduce_perf
-rwxr-xr-x 1 root root 18237864  5月  2 20:24 alltoall_perf
-rwxr-xr-x 1 root root 18234488  5月  2 20:24 broadcast_perf
-rwxr-xr-x 1 root root 18239912  5月  2 20:24 gather_perf
-rwxr-xr-x 1 root root 18240816  5月  2 20:24 hypercube_perf
-rwxr-xr-x 1 root root 18234704  5月  2 20:24 reduce_perf
-rwxr-xr-x 1 root root 18235432  5月  2 20:24 reduce_scatter_perf
-rwxr-xr-x 1 root root 18238752  5月  2 20:24 scatter_perf
-rwxr-xr-x 1 root root 18239248  5月  2 20:24 sendrecv_perf
-rw-r--r-- 1 root root     5096  5月  2 19:59 timer.o
drwxr-xr-x 2 root root        3  5月  2 20:24 verifiable

AI:
太好了！编译非常顺利，所有的NCCL测试程序都已成功编译。现在您已经准备好运行NCCL测试了。

接下来，让我们创建一个脚本来测试NCCL的性能：

现在我们已经成功编译了NCCL测试，让我们运行一些基本的测试来验证一切工作正常。请执行以下命令：

```bash
chmod +x run_nccl_tests.sh
./run_nccl_tests.sh
```

USER:
    内容参考：run_nccl_test_result.md

AI:
看来您的测试运行遇到了一个 "invalid device ordinal" 错误。这个错误通常表示进程无法访问指定的GPU。您有2个GPU，但测试进程无法正确地访问这些GPU。

让我们修改测试脚本来解决这个问题：

我们遇到的问题是"invalid device ordinal"，这表明测试进程无法正确访问GPU。我创建了一个新脚本，尝试几种不同的方法来运行NCCL测试：

1. 设置重要的NCCL环境变量，如禁用IB通信、启用P2P和共享内存
2. 明确设置`CUDA_VISIBLE_DEVICES`来指定可见的GPU
3. 尝试单进程模式（直接运行可执行文件而不是通过MPI）
4. 如果单进程模式失败，逐个测试每个GPU

请执行以下命令：

```bash
chmod +x fix_nccl_tests.sh
./fix_nccl_tests.sh
```

执行后，请将输出结果反馈给我，我会根据结果进行进一步调整。


USER:
    内容参考：fix_nccl_tests_result.md

AI:
    太好了！单进程模式成功运行，这表明您的NCCL配置现在可以正常工作了。从输出中可以看到：

1. 两个GPU (RTX 4090) 都被正确识别和使用
2. NCCL通过本地SHM (共享内存) 在两个GPU之间建立了连接
3. 已成功运行all_reduce测试，并获得了性能数据

现在让我们创建一个更全面的测试脚本，包括更多的NCCL操作和更详细的配置选项：

非常好！我们已经成功解决了NCCL测试的问题。从上一次的输出中可以看到，两个GPU之间已经成功建立了通信，并且all_reduce性能测试运行正常。

现在我创建了一个更全面的测试脚本，它将运行多种NCCL集体通信操作。这些测试将帮助您评估您的多GPU系统在不同通信模式下的性能。请执行以下命令：

```bash
chmod +x comprehensive_nccl_tests.sh
./comprehensive_nccl_tests.sh
```

这个脚本会：
1. 运行4种主要的NCCL通信原语测试：all_reduce, broadcast, all_gather, reduce_scatter
2. 对每种测试使用从8字节到128MB的数据大小范围
3. 提供一个测试总结和性能调优提示

如果您想要更全面的测试，可以在脚本中取消注释其他测试类型。

执行后，您将得到一个完整的NCCL功能和性能报告。您可以根据需要反馈结果，或者修改脚本中的参数进行更详细的测试。

USER:
    内容参考：comprehensive_nccl_tests_results.md


总结：


---

## 🎯 核心结论（先说重点）

> **你的多卡通信是走 PCIe 总线 + SHM（Shared Memory）方式，没有使用 NVLink，因此带宽受限在 \~1-2 GB/s，是正常表现。但对于大模型训练（尤其是参数同步频繁的任务），会成为瓶颈。**

---

## 🧠 深度分析

### 1. ✅ GPU 连接架构分析

* **GPU 型号**：RTX 4090（AD102 芯片）本身**不具备 NVLink 支持**；
* **PCIe 拓扑**：

  ```
  GPU 0: 0000:01:00.0
  GPU 1: 0000:81:00.0
  ```

  说明这两块卡分别挂在不同的 CPU root port 下（通常表示横跨 NUMA 节点或根桥分离）；
* **通信路径：SHM/direct/direct**：

  * 属于 NCCL 自动 fallback 到 CPU-hosted memory + PCIe；
  * 本质是通过共享内存或 DMA 拷贝完成通信；
  * 没有 NVLink 或 RDMA 协议支持。

---

### 2. 📉 性能数据解释

| 操作类型          | 峰值带宽           | 说明                   |
| ------------- | -------------- | -------------------- |
| AllReduce     | \~0.94 GB/s    | 正常（PCIe 限制）          |
| Broadcast     | \~1.0 GB/s     | 接近极限                 |
| AllGather     | \~2.3 GB/s     | 较好，因为带宽可累加           |
| ReduceScatter | \~1.4 GB/s     | 也在合理范围               |
| **平均总带宽**     | \~0.4-0.5 GB/s | 属于典型 PCIe-only 架构下表现 |

> ❗对比参考：
>
> * **NVLink 2.0**：带宽为 50 GB/s per link (双向)；
> * **PCIe Gen4 x16**：带宽理论上是 \~32 GB/s，**但你实际观测到的带宽只有 1-2 GB/s**，这说明是 host 内存拷贝 + buffer 层的限制。

---

### 3. 🔄 数据规模与通信效率关系

你也正确观察到：

* 小于 32KB 的小包传输非常低效（高开销 + kernel call 等待）；
* 数据量越大（如 8MB \~ 128MB），通信性能越趋于稳定；
* 建议训练中避免频繁的小量同步。

---

### 4. ⚙️ GPU 利用率异常说明

你提到：

> GPU0 利用率 30%，GPU1 为 0%

可能的原因：

* NCCL 的 `rank 0` 默认是 initiator（发送发起端），而 `rank 1` 更多是 passive 接收者；
* NCCL benchmarks 本身测试不代表训练负载，它在一些模式下并不会满载所有 GPU；
* 你可以用 `nvidia-smi dmon` 或 `nsight-sys` 做动态带宽与使用率观测，获取更真实的训练表现。

---

## ✅ 你的配置表现是否合理？

是的，你的系统表现符合 RTX 4090 + PCIe 架构的正常预期，没有 NVLink 就意味着：

* NCCL 通信主要走 host → device 拷贝；
* 每秒有效带宽在 0.5\~2 GB/s；
* 更大 buffer 更稳定；多小包开销陡增。

---

## 📌 建议总结

| 项目      | 建议                                                                   |
| ------- | -------------------------------------------------------------------- |
| 通信架构    | RTX 4090 没有 NVLink → 不建议用于大量 AllReduce 的多卡训练（如原生 FP32 DDP）           |
| 优化策略    | 使用 ZeRO、LoRA 等减少通信压力；batch 大些合并通信；或用 gradient accumulation           |
| 硬件升级    | 若高通信负载是常态：使用 A100/H100 系列（带 NVLink/NVSwitch）或 InfiniBand 多节点         |
| NCCL 参数 | 你提到的 `NCCL_BUFFSIZE`、`NCCL_ALGO` 等都可以尝试，如 `NCCL_ALGO=Tree` 在小模型场景更高效 |

