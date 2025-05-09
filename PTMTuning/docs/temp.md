### SFT DATA
- Synthetic 
- Rejection Sampling 



已思考若干秒


下面是对当前 `train_sft.py` 脚本主要功能模块的拆解说明：

---

## 1. 输入（Input）

* **配置**：通过 Hydra 从 `conf/conf_abl_semi.yaml` 中读取

  * `cfg.model.name_or_path`：基准大模型路径或名字
  * `cfg.model.special_tokens`：自定义的 special tokens
  * `cfg.data.samples`：训练样本列表，每项包含：

    * `question`（题干文本）
    * `correct`（正确选项）
    * `incorrect`（错误选项）
    * `example_cots`（可选的示例 CoT list，每个含 `question`/`incorrect`/`reason`）
    * `cot`（闭源模型生成的思考链）
    * `misconception`（最终要学的短句标签）
  * `cfg.data.prompt_template` 或 `cfg.data.prompt_templates`：一个或多个格式化模板字符串
  * `cfg.train.*`：训练超参，如 `batch_size`、`epochs`、`lr`、`mixed_precision`、`deepspeed` 配置等

* **样本预处理**

  * 在 `CotDataset.__getitem__` 中，将上面的字段按模板拼接成 `prompt`，再加上 `<COT_START>`/`<COT_END>` 及原模型输出的 `cot` 与 `misconception`，构成完整的输入 (`prompt + cot + final`) 和标签。

---

## 2. 输出（Output）

* **模型检查点 &日志**

  * SFTTrainer 自带的输出目录下会保存：

    * 训练日志（如学习率、loss、步数）
    * 最终的 fine-tuned 模型权重
* **Loss 曲线图**

  * 脚本末尾会把训练过程中累积的 `loss` 数组绘制成 `loss_curve.png`，存放在当前工作目录。

---

## 3. 数据构成（Data Composition）

* **Prompt 部分**：

  * 使用一个或多个 `prompt_template` 填充题干、选项、示例 CoT
* **标签部分**：

  * 先是闭源模型的思考链 `cot`，后接简短的 `misconception` 句子
* **Loss Mask**：

  * 对输入中的 `prompt` 区段，通过把相应 `labels` 值设为 `-100`，在计算 CrossEntropy 时完全忽略 prompt 部分，只对模型生成的 CoT+misconception 计算 loss。

---

## 4. 训练框架（Training Framework）

* **Transformers + TRL**

  * 使用 HuggingFace `AutoModelForCausalLM` + `AutoTokenizer`
  * 通过 `trl.SFTTrainer` 与 `trl.SFTTrainingArguments` 封装训练流程
* **DeepSpeed Plugin**

  * 在 `SFTTrainingArguments(deepspeed=cfg.train.deepspeed)` 中自动启用 DeepSpeed
  * 支持 ZeRO、FP16 等分布式加速与显存优化
* **Hydra**

  * 全局配置管理，支持命令行与配置文件混合覆盖超参

---

## 5. 代码功能构成（Module Breakdown）

1. **`plot_loss`**

   * 接收 step-wise loss 列表，绘制并保存训练曲线图

2. **`CotDataset`**

   * 自定义 `Dataset` 类
   * 负责：模板抽样、prompt 拼接、tokenize、构造 `input_ids`、`attention_mask`、`labels`（并遮盖 prompt 部分）

3. **主函数 `main`**

   * **初始化**：加载 tokenizer、添加 special tokens
   * **模型加载**：`AutoModelForCausalLM.from_pretrained` + `resize_token_embeddings`
   * **数据准备**：用 `CotDataset` 生成训练集
   * **训练参数**：`SFTTrainingArguments` 中配置 Deepspeed、FP16、batch、epoch 等
   * **Trainer**：`SFTTrainer` 负责训练循环、梯度累积、日志记录
   * **结果处理**：训练完成后收集 `result.history['loss']` 并调用 `plot_loss`

---

## 6. 分布式相关（Distributed Training）

* **DeepSpeed**：

  * 由 `SFTTrainingArguments(deepspeed=…)` 自动注入
  * 支持多 GPU ZeRO 优化、梯度分片、通信重叠等
* **Mixed Precision**：

  * 通过 `fp16=cfg.train.mixed_precision=='fp16'` 启用 AMP
* **Hydra + Accelerate**：

  * 虽未显式在代码中调用 `Accelerator`，但 `SFTTrainer` 底层会使用 Accelerate 和 DeepSpeed，自动管理多进程分布式环境

---

这样，你可以对照以上六大块来快速了解脚本里各模块的职责、数据流向和并行化加速方式。
