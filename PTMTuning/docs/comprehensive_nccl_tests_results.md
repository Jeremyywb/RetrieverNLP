(myproject_env) (base) root@0fed198eb8ba:/home# chmod +x comprehensive_nccl_tests.sh
./comprehensive_nccl_tests.sh
运行全面NCCL测试...
检测到 2 个GPU
=================== 系统信息 ===================
Fri May  2 20:30:30 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
| 31%   42C    P2            137W /  450W |   20594MiB /  24564MiB |     30%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        On  |   00000000:81:00.0 Off |                  Off |
| 37%   45C    P2            109W /  450W |   20594MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
===============================================
使用CUDA_VISIBLE_DEVICES=0,1
运行 all_reduce 测试...
--------------------------------------
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 528157 on 0fed198eb8ba device  0 [0000:01:00] NVIDIA GeForce RTX 4090
#  Rank  1 Group  0 Pid 528157 on 0fed198eb8ba device  1 [0000:81:00] NVIDIA GeForce RTX 4090
0fed198eb8ba:528157:528157 [0] NCCL INFO Bootstrap : Using eth0:10.60.10.2<0>
0fed198eb8ba:528157:528157 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
0fed198eb8ba:528157:528157 [0] NCCL INFO NET/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-net.so
0fed198eb8ba:528157:528157 [0] NCCL INFO NET/Plugin: Using internal network plugin.
0fed198eb8ba:528157:528157 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.21.5+cuda12.4
0fed198eb8ba:528157:528268 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
0fed198eb8ba:528157:528268 [0] NCCL INFO NET/Socket : Using [0]eth0:10.60.10.2<0>
0fed198eb8ba:528157:528268 [0] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:528157:528268 [0] NCCL INFO Using network Socket
0fed198eb8ba:528157:528269 [1] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:528157:528269 [1] NCCL INFO Using network Socket
0fed198eb8ba:528157:528269 [1] NCCL INFO ncclCommInitRank comm 0x55d8db257ea0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xc4cf3e318640a470 - Init START
0fed198eb8ba:528157:528268 [0] NCCL INFO ncclCommInitRank comm 0x55d8db252140 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xc4cf3e318640a470 - Init START
0fed198eb8ba:528157:528269 [1] NCCL INFO NCCL_SHM_DISABLE set by environment to 0.
0fed198eb8ba:528157:528269 [1] NCCL INFO Setting affinity for GPU 1 to 07,00000000,00000007,00000000
0fed198eb8ba:528157:528268 [0] NCCL INFO Setting affinity for GPU 0 to f000007f,00000000,f000007f
0fed198eb8ba:528157:528268 [0] NCCL INFO comm 0x55d8db252140 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
0fed198eb8ba:528157:528268 [0] NCCL INFO Channel 00/02 :    0   1
0fed198eb8ba:528157:528269 [1] NCCL INFO comm 0x55d8db257ea0 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
0fed198eb8ba:528157:528268 [0] NCCL INFO Channel 01/02 :    0   1
0fed198eb8ba:528157:528269 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0
0fed198eb8ba:528157:528269 [1] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:528157:528268 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
0fed198eb8ba:528157:528268 [0] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:528157:528268 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:528157:528269 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:528157:528268 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:528157:528269 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:528157:528268 [0] NCCL INFO Connected all rings
0fed198eb8ba:528157:528268 [0] NCCL INFO Connected all trees
0fed198eb8ba:528157:528268 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:528157:528268 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:528157:528269 [1] NCCL INFO Connected all rings
0fed198eb8ba:528157:528269 [1] NCCL INFO Connected all trees
0fed198eb8ba:528157:528269 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:528157:528269 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:528157:528268 [0] NCCL INFO TUNER/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-tuner.so
0fed198eb8ba:528157:528268 [0] NCCL INFO TUNER/Plugin: Using internal tuner plugin.
0fed198eb8ba:528157:528268 [0] NCCL INFO ncclCommInitRank comm 0x55d8db252140 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xc4cf3e318640a470 - Init COMPLETE
0fed198eb8ba:528157:528269 [1] NCCL INFO ncclCommInitRank comm 0x55d8db257ea0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xc4cf3e318640a470 - Init COMPLETE
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    14.54    0.00    0.00      0    15.94    0.00    0.00      0
          16             4     float     sum      -1    16.82    0.00    0.00      0    16.50    0.00    0.00      0
          32             8     float     sum      -1    21.87    0.00    0.00      0    22.93    0.00    0.00      0
          64            16     float     sum      -1    15.60    0.00    0.00      0    205.8    0.00    0.00      0
         128            32     float     sum      -1    16.86    0.01    0.01      0    21.08    0.01    0.01      0
         256            64     float     sum      -1    14.14    0.02    0.02      0    15.77    0.02    0.02      0
         512           128     float     sum      -1    21.16    0.02    0.02      0    19.78    0.03    0.03      0
        1024           256     float     sum      -1    17.06    0.06    0.06      0    24.83    0.04    0.04      0
        2048           512     float     sum      -1    24.02    0.09    0.09      0    16.07    0.13    0.13      0
        4096          1024     float     sum      -1    25.03    0.16    0.16      0    16.21    0.25    0.25      0
        8192          2048     float     sum      -1    17.63    0.46    0.46      0    27.96    0.29    0.29      0
       16384          4096     float     sum      -1    23.77    0.69    0.69      0    30.47    0.54    0.54      0
       32768          8192     float     sum      -1    49.76    0.66    0.66      0    50.34    0.65    0.65      0
       65536         16384     float     sum      -1    87.27    0.75    0.75      0    82.63    0.79    0.79      0
      131072         32768     float     sum      -1    243.1    0.54    0.54      0    263.1    0.50    0.50      0
      262144         65536     float     sum      -1    264.4    0.99    0.99      0    424.9    0.62    0.62      0
      524288        131072     float     sum      -1   1302.4    0.40    0.40      0    987.9    0.53    0.53      0
     1048576        262144     float     sum      -1   2048.2    0.51    0.51      0   1684.4    0.62    0.62      0
     2097152        524288     float     sum      -1   3227.3    0.65    0.65      0   2824.5    0.74    0.74      0
     4194304       1048576     float     sum      -1   5047.3    0.83    0.83      0   5010.0    0.84    0.84      0
     8388608       2097152     float     sum      -1   9646.6    0.87    0.87      0   9124.1    0.92    0.92      0
    16777216       4194304     float     sum      -1    18562    0.90    0.90      0    17809    0.94    0.94      0
    33554432       8388608     float     sum      -1    36875    0.91    0.91      0    35974    0.93    0.93      0
    67108864      16777216     float     sum      -1    75422    0.89    0.89      0    73478    0.91    0.91      0
   134217728      33554432     float     sum      -1   150405    0.89    0.89      0   148570    0.90    0.90      0
0fed198eb8ba:528157:528157 [1] NCCL INFO comm 0x55d8db252140 rank 0 nranks 2 cudaDev 0 busId 1000 - Destroy COMPLETE
0fed198eb8ba:528157:528157 [1] NCCL INFO comm 0x55d8db257ea0 rank 1 nranks 2 cudaDev 1 busId 81000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.450547 
#

--------------------------------------
all_reduce 测试完成

运行 broadcast 测试...
--------------------------------------
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 530158 on 0fed198eb8ba device  0 [0000:01:00] NVIDIA GeForce RTX 4090
#  Rank  1 Group  0 Pid 530158 on 0fed198eb8ba device  1 [0000:81:00] NVIDIA GeForce RTX 4090
0fed198eb8ba:530158:530158 [0] NCCL INFO Bootstrap : Using eth0:10.60.10.2<0>
0fed198eb8ba:530158:530158 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
0fed198eb8ba:530158:530158 [0] NCCL INFO NET/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-net.so
0fed198eb8ba:530158:530158 [0] NCCL INFO NET/Plugin: Using internal network plugin.
0fed198eb8ba:530158:530158 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.21.5+cuda12.4
0fed198eb8ba:530158:530324 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
0fed198eb8ba:530158:530324 [0] NCCL INFO NET/Socket : Using [0]eth0:10.60.10.2<0>
0fed198eb8ba:530158:530324 [0] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:530158:530324 [0] NCCL INFO Using network Socket
0fed198eb8ba:530158:530325 [1] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:530158:530325 [1] NCCL INFO Using network Socket
0fed198eb8ba:530158:530324 [0] NCCL INFO ncclCommInitRank comm 0x55a3ef1d03a0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xbcbbac5a562c621e - Init START
0fed198eb8ba:530158:530325 [1] NCCL INFO ncclCommInitRank comm 0x55a3ef1d6100 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xbcbbac5a562c621e - Init START
0fed198eb8ba:530158:530324 [0] NCCL INFO NCCL_SHM_DISABLE set by environment to 0.
0fed198eb8ba:530158:530324 [0] NCCL INFO Setting affinity for GPU 0 to f000007f,00000000,f000007f
0fed198eb8ba:530158:530325 [1] NCCL INFO Setting affinity for GPU 1 to 07,00000000,00000007,00000000
0fed198eb8ba:530158:530324 [0] NCCL INFO comm 0x55a3ef1d03a0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
0fed198eb8ba:530158:530325 [1] NCCL INFO comm 0x55a3ef1d6100 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
0fed198eb8ba:530158:530324 [0] NCCL INFO Channel 00/02 :    0   1
0fed198eb8ba:530158:530325 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0
0fed198eb8ba:530158:530325 [1] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:530158:530324 [0] NCCL INFO Channel 01/02 :    0   1
0fed198eb8ba:530158:530324 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
0fed198eb8ba:530158:530324 [0] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:530158:530324 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:530158:530325 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:530158:530324 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:530158:530325 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:530158:530324 [0] NCCL INFO Connected all rings
0fed198eb8ba:530158:530324 [0] NCCL INFO Connected all trees
0fed198eb8ba:530158:530324 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:530158:530324 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:530158:530325 [1] NCCL INFO Connected all rings
0fed198eb8ba:530158:530325 [1] NCCL INFO Connected all trees
0fed198eb8ba:530158:530325 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:530158:530325 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:530158:530324 [0] NCCL INFO TUNER/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-tuner.so
0fed198eb8ba:530158:530324 [0] NCCL INFO TUNER/Plugin: Using internal tuner plugin.
0fed198eb8ba:530158:530324 [0] NCCL INFO ncclCommInitRank comm 0x55a3ef1d03a0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xbcbbac5a562c621e - Init COMPLETE
0fed198eb8ba:530158:530325 [1] NCCL INFO ncclCommInitRank comm 0x55a3ef1d6100 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xbcbbac5a562c621e - Init COMPLETE
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float    none       0    16.26    0.00    0.00      0    15.55    0.00    0.00      0
          16             4     float    none       0    13.87    0.00    0.00      0    12.96    0.00    0.00      0
          32             8     float    none       0    20.28    0.00    0.00      0    38.64    0.00    0.00      0
          64            16     float    none       0    17.62    0.00    0.00      0    26.60    0.00    0.00      0
         128            32     float    none       0    24.37    0.01    0.01      0    19.84    0.01    0.01      0
         256            64     float    none       0    18.72    0.01    0.01      0    15.22    0.02    0.02      0
         512           128     float    none       0    21.54    0.02    0.02      0    15.67    0.03    0.03      0
        1024           256     float    none       0    15.79    0.06    0.06      0    30.70    0.03    0.03      0
        2048           512     float    none       0    17.81    0.11    0.11      0    17.25    0.12    0.12      0
        4096          1024     float    none       0    30.08    0.14    0.14      0    14.13    0.29    0.29      0
        8192          2048     float    none       0    21.92    0.37    0.37      0    15.48    0.53    0.53      0
       16384          4096     float    none       0    31.04    0.53    0.53      0    35.90    0.46    0.46      0
       32768          8192     float    none       0    62.69    0.52    0.52      0    61.82    0.53    0.53      0
       65536         16384     float    none       0    189.5    0.35    0.35      0    195.1    0.34    0.34      0
      131072         32768     float    none       0    196.2    0.67    0.67      0    195.4    0.67    0.67      0
      262144         65536     float    none       0    383.4    0.68    0.68      0    371.4    0.71    0.71      0
      524288        131072     float    none       0    776.4    0.68    0.68      0    774.9    0.68    0.68      0
     1048576        262144     float    none       0   1419.4    0.74    0.74      0   1429.4    0.73    0.73      0
     2097152        524288     float    none       0   2474.1    0.85    0.85      0   2827.0    0.74    0.74      0
     4194304       1048576     float    none       0   5223.9    0.80    0.80      0   5321.3    0.79    0.79      0
     8388608       2097152     float    none       0    10356    0.81    0.81      0   9412.3    0.89    0.89      0
    16777216       4194304     float    none       0    21334    0.79    0.79      0    21634    0.78    0.78      0
    33554432       8388608     float    none       0    44011    0.76    0.76      0    39885    0.84    0.84      0
    67108864      16777216     float    none       0    73624    0.91    0.91      0    67305    1.00    1.00      0
   134217728      33554432     float    none       0   139269    0.96    0.96      0   147567    0.91    0.91      0
0fed198eb8ba:530158:530158 [1] NCCL INFO comm 0x55a3ef1d03a0 rank 0 nranks 2 cudaDev 0 busId 1000 - Destroy COMPLETE
0fed198eb8ba:530158:530158 [1] NCCL INFO comm 0x55a3ef1d6100 rank 1 nranks 2 cudaDev 1 busId 81000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.43744 
#

--------------------------------------
broadcast 测试完成

运行 all_gather 测试...
--------------------------------------
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 532235 on 0fed198eb8ba device  0 [0000:01:00] NVIDIA GeForce RTX 4090
#  Rank  1 Group  0 Pid 532235 on 0fed198eb8ba device  1 [0000:81:00] NVIDIA GeForce RTX 4090
0fed198eb8ba:532235:532235 [0] NCCL INFO Bootstrap : Using eth0:10.60.10.2<0>
0fed198eb8ba:532235:532235 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
0fed198eb8ba:532235:532235 [0] NCCL INFO NET/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-net.so
0fed198eb8ba:532235:532235 [0] NCCL INFO NET/Plugin: Using internal network plugin.
0fed198eb8ba:532235:532235 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.21.5+cuda12.4
0fed198eb8ba:532235:532397 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
0fed198eb8ba:532235:532397 [0] NCCL INFO NET/Socket : Using [0]eth0:10.60.10.2<0>
0fed198eb8ba:532235:532397 [0] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:532235:532397 [0] NCCL INFO Using network Socket
0fed198eb8ba:532235:532398 [1] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:532235:532398 [1] NCCL INFO Using network Socket
0fed198eb8ba:532235:532398 [1] NCCL INFO ncclCommInitRank comm 0x5630bb964030 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xc3914b608739e848 - Init START
0fed198eb8ba:532235:532397 [0] NCCL INFO ncclCommInitRank comm 0x5630bb95e2d0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xc3914b608739e848 - Init START
0fed198eb8ba:532235:532398 [1] NCCL INFO NCCL_SHM_DISABLE set by environment to 0.
0fed198eb8ba:532235:532398 [1] NCCL INFO Setting affinity for GPU 1 to 07,00000000,00000007,00000000
0fed198eb8ba:532235:532397 [0] NCCL INFO Setting affinity for GPU 0 to f000007f,00000000,f000007f
0fed198eb8ba:532235:532397 [0] NCCL INFO comm 0x5630bb95e2d0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
0fed198eb8ba:532235:532397 [0] NCCL INFO Channel 00/02 :    0   1
0fed198eb8ba:532235:532397 [0] NCCL INFO Channel 01/02 :    0   1
0fed198eb8ba:532235:532398 [1] NCCL INFO comm 0x5630bb964030 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
0fed198eb8ba:532235:532397 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
0fed198eb8ba:532235:532397 [0] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:532235:532398 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0
0fed198eb8ba:532235:532398 [1] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:532235:532397 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:532235:532398 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:532235:532397 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:532235:532398 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:532235:532397 [0] NCCL INFO Connected all rings
0fed198eb8ba:532235:532397 [0] NCCL INFO Connected all trees
0fed198eb8ba:532235:532398 [1] NCCL INFO Connected all rings
0fed198eb8ba:532235:532398 [1] NCCL INFO Connected all trees
0fed198eb8ba:532235:532397 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:532235:532398 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:532235:532398 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:532235:532397 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:532235:532397 [0] NCCL INFO TUNER/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-tuner.so
0fed198eb8ba:532235:532397 [0] NCCL INFO TUNER/Plugin: Using internal tuner plugin.
0fed198eb8ba:532235:532397 [0] NCCL INFO ncclCommInitRank comm 0x5630bb95e2d0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xc3914b608739e848 - Init COMPLETE
0fed198eb8ba:532235:532398 [1] NCCL INFO ncclCommInitRank comm 0x5630bb964030 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0xc3914b608739e848 - Init COMPLETE
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           0             0     float    none      -1     0.56    0.00    0.00      0     0.53    0.00    0.00      0
           0             0     float    none      -1     0.52    0.00    0.00      0     0.52    0.00    0.00      0
          32             4     float    none      -1    12.86    0.00    0.00      0    18.82    0.00    0.00      0
          64             8     float    none      -1    19.39    0.00    0.00      0    13.95    0.00    0.00      0
         128            16     float    none      -1    13.16    0.01    0.00      0    14.10    0.01    0.00      0
         256            32     float    none      -1    11.60    0.02    0.01      0    18.79    0.01    0.01      0
         512            64     float    none      -1    14.72    0.03    0.02      0    13.31    0.04    0.02      0
        1024           128     float    none      -1    14.26    0.07    0.04      0    12.49    0.08    0.04      0
        2048           256     float    none      -1    20.99    0.10    0.05      0    13.19    0.16    0.08      0
        4096           512     float    none      -1    14.89    0.28    0.14      0    18.80    0.22    0.11      0
        8192          1024     float    none      -1    18.13    0.45    0.23      0    16.62    0.49    0.25      0
       16384          2048     float    none      -1    24.69    0.66    0.33      0    24.67    0.66    0.33      0
       32768          4096     float    none      -1    38.32    0.86    0.43      0    42.85    0.76    0.38      0
       65536          8192     float    none      -1    74.11    0.88    0.44      0    67.70    0.97    0.48      0
      131072         16384     float    none      -1    183.7    0.71    0.36      0    194.6    0.67    0.34      0
      262144         32768     float    none      -1    181.9    1.44    0.72      0    187.7    1.40    0.70      0
      524288         65536     float    none      -1    326.8    1.60    0.80      0    320.6    1.64    0.82      0
     1048576        131072     float    none      -1    636.9    1.65    0.82      0    661.7    1.58    0.79      0
     2097152        262144     float    none      -1   1244.5    1.69    0.84      0   1186.5    1.77    0.88      0
     4194304        524288     float    none      -1   2454.8    1.71    0.85      0   2412.4    1.74    0.87      0
     8388608       1048576     float    none      -1   4934.2    1.70    0.85      0   4871.5    1.72    0.86      0
    16777216       2097152     float    none      -1   9753.9    1.72    0.86      0   9385.6    1.79    0.89      0
    33554432       4194304     float    none      -1    18956    1.77    0.89      0    18880    1.78    0.89      0
    67108864       8388608     float    none      -1    34189    1.96    0.98      0    32588    2.06    1.03      0
   134217728      16777216     float    none      -1    79658    1.68    0.84      0    57809    2.32    1.16      0
0fed198eb8ba:532235:532235 [1] NCCL INFO comm 0x5630bb95e2d0 rank 0 nranks 2 cudaDev 0 busId 1000 - Destroy COMPLETE
0fed198eb8ba:532235:532235 [1] NCCL INFO comm 0x5630bb964030 rank 1 nranks 2 cudaDev 1 busId 81000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.428849 
#

--------------------------------------
all_gather 测试完成

运行 reduce_scatter 测试...
--------------------------------------
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 533436 on 0fed198eb8ba device  0 [0000:01:00] NVIDIA GeForce RTX 4090
#  Rank  1 Group  0 Pid 533436 on 0fed198eb8ba device  1 [0000:81:00] NVIDIA GeForce RTX 4090
0fed198eb8ba:533436:533436 [0] NCCL INFO Bootstrap : Using eth0:10.60.10.2<0>
0fed198eb8ba:533436:533436 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
0fed198eb8ba:533436:533436 [0] NCCL INFO NET/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-net.so
0fed198eb8ba:533436:533436 [0] NCCL INFO NET/Plugin: Using internal network plugin.
0fed198eb8ba:533436:533436 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.21.5+cuda12.4
0fed198eb8ba:533436:533598 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
0fed198eb8ba:533436:533598 [0] NCCL INFO NET/Socket : Using [0]eth0:10.60.10.2<0>
0fed198eb8ba:533436:533598 [0] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:533436:533598 [0] NCCL INFO Using network Socket
0fed198eb8ba:533436:533599 [1] NCCL INFO Using non-device net plugin version 0
0fed198eb8ba:533436:533599 [1] NCCL INFO Using network Socket
0fed198eb8ba:533436:533599 [1] NCCL INFO ncclCommInitRank comm 0x55d51933dd90 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0x74d4d09b92596c09 - Init START
0fed198eb8ba:533436:533598 [0] NCCL INFO ncclCommInitRank comm 0x55d519338030 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0x74d4d09b92596c09 - Init START
0fed198eb8ba:533436:533599 [1] NCCL INFO NCCL_SHM_DISABLE set by environment to 0.
0fed198eb8ba:533436:533599 [1] NCCL INFO Setting affinity for GPU 1 to 07,00000000,00000007,00000000
0fed198eb8ba:533436:533598 [0] NCCL INFO Setting affinity for GPU 0 to f000007f,00000000,f000007f
0fed198eb8ba:533436:533598 [0] NCCL INFO comm 0x55d519338030 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
0fed198eb8ba:533436:533598 [0] NCCL INFO Channel 00/02 :    0   1
0fed198eb8ba:533436:533598 [0] NCCL INFO Channel 01/02 :    0   1
0fed198eb8ba:533436:533598 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
0fed198eb8ba:533436:533599 [1] NCCL INFO comm 0x55d51933dd90 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
0fed198eb8ba:533436:533598 [0] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:533436:533599 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0
0fed198eb8ba:533436:533599 [1] NCCL INFO P2P Chunksize set to 131072
0fed198eb8ba:533436:533598 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:533436:533599 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:533436:533598 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
0fed198eb8ba:533436:533599 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
0fed198eb8ba:533436:533599 [1] NCCL INFO Connected all rings
0fed198eb8ba:533436:533599 [1] NCCL INFO Connected all trees
0fed198eb8ba:533436:533599 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:533436:533599 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:533436:533598 [0] NCCL INFO Connected all rings
0fed198eb8ba:533436:533598 [0] NCCL INFO Connected all trees
0fed198eb8ba:533436:533598 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
0fed198eb8ba:533436:533598 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
0fed198eb8ba:533436:533598 [0] NCCL INFO TUNER/Plugin: Plugin load returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory : when loading libnccl-tuner.so
0fed198eb8ba:533436:533598 [0] NCCL INFO TUNER/Plugin: Using internal tuner plugin.
0fed198eb8ba:533436:533598 [0] NCCL INFO ncclCommInitRank comm 0x55d519338030 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0x74d4d09b92596c09 - Init COMPLETE
0fed198eb8ba:533436:533599 [1] NCCL INFO ncclCommInitRank comm 0x55d51933dd90 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 81000 commId 0x74d4d09b92596c09 - Init COMPLETE
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           0             0     float     sum      -1     0.53    0.00    0.00      0     0.53    0.00    0.00      0
           0             0     float     sum      -1     0.53    0.00    0.00      0     0.53    0.00    0.00      0
          32             4     float     sum      -1    18.17    0.00    0.00      0    14.56    0.00    0.00      0
          64             8     float     sum      -1    13.78    0.00    0.00      0    21.18    0.00    0.00      0
         128            16     float     sum      -1    15.95    0.01    0.00      0    15.93    0.01    0.00      0
         256            32     float     sum      -1    22.50    0.01    0.01      0    16.84    0.02    0.01      0
         512            64     float     sum      -1    19.48    0.03    0.01      0    24.26    0.02    0.01      0
        1024           128     float     sum      -1    20.66    0.05    0.02      0    17.19    0.06    0.03      0
        2048           256     float     sum      -1    18.26    0.11    0.06      0    18.36    0.11    0.06      0
        4096           512     float     sum      -1    21.31    0.19    0.10      0    17.38    0.24    0.12      0
        8192          1024     float     sum      -1    18.82    0.44    0.22      0    29.02    0.28    0.14      0
       16384          2048     float     sum      -1    35.13    0.47    0.23      0    25.65    0.64    0.32      0
       32768          4096     float     sum      -1    39.14    0.84    0.42      0    39.27    0.83    0.42      0
       65536          8192     float     sum      -1    68.77    0.95    0.48      0    74.61    0.88    0.44      0
      131072         16384     float     sum      -1    197.0    0.67    0.33      0    230.9    0.57    0.28      0
      262144         32768     float     sum      -1    234.4    1.12    0.56      0    302.4    0.87    0.43      0
      524288         65536     float     sum      -1    403.5    1.30    0.65      0    721.7    0.73    0.36      0
     1048576        131072     float     sum      -1   1145.6    0.92    0.46      0   1232.0    0.85    0.43      0
     2097152        262144     float     sum      -1   1798.0    1.17    0.58      0   1683.0    1.25    0.62      0
     4194304        524288     float     sum      -1   2903.6    1.44    0.72      0   2957.2    1.42    0.71      0
     8388608       1048576     float     sum      -1   5786.7    1.45    0.72      0   5905.4    1.42    0.71      0
    16777216       2097152     float     sum      -1    11706    1.43    0.72      0    11351    1.48    0.74      0
    33554432       4194304     float     sum      -1    23364    1.44    0.72      0    23251    1.44    0.72      0
    67108864       8388608     float     sum      -1    47590    1.41    0.71      0    46914    1.43    0.72      0
   134217728      16777216     float     sum      -1    97895    1.37    0.69      0    97113    1.38    0.69      0
0fed198eb8ba:533436:533436 [1] NCCL INFO comm 0x55d519338030 rank 0 nranks 2 cudaDev 0 busId 1000 - Destroy COMPLETE
0fed198eb8ba:533436:533436 [1] NCCL INFO comm 0x55d51933dd90 rank 1 nranks 2 cudaDev 1 busId 81000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.327287 
#

--------------------------------------
reduce_scatter 测试完成

所有测试完成！

================= NCCL测试总结 =================
GPU数量: 2
环境变量:
  NCCL_DEBUG=INFO
  NCCL_P2P_DISABLE=0
  NCCL_SHM_DISABLE=0
  NCCL_IB_DISABLE=1
  CUDA_VISIBLE_DEVICES=0,1

性能提示:
1. 若要获得更好的性能，请确保GPU之间有直接的PCIe或NVLink连接
2. 对于大型集群，可能需要配置网络接口和调整NCCL参数
3. 您可以使用以下环境变量调优NCCL性能:
   - NCCL_SOCKET_IFNAME: 指定网络接口
   - NCCL_BUFFSIZE: 调整缓冲区大小
   - NCCL_RINGS: 配置NCCL通信环
   - NCCL_ALGO: 设置集体通信算法