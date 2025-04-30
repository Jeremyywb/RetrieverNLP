# 高效的 LoRA 与非 LoRA 模型管理实现

设计了一个清晰、更高效的实现方案，解决了多阶段训练模型路径管理问题。这个方案有以下几个核心优点：

1. **清晰的路径管理策略**：根据是否使用 LoRA 自动选择合适的路径
2. **统一的保存逻辑**：任务特定模型与基础模型分离存储
3. **安全的参数过滤**：避免 BERT 模型中的 `labels` 参数问题
4. **正确的 LoRA 配置**：使用 `TaskType.FEATURE_EXTRACTION` 而非 `CAUSAL_LM`
5. **健壮的错误处理**：处理模型加载失败的情况

## 核心设计思路

### LoRA 模式下的路径管理
- 基础模型保存在公共路径 `base_backbone_path`
- LoRA 适配器和 head 模型保存在任务特定路径 `output_dir_{task_name}`
- LoRA 适配器保存在 `output_dir_{task_name}_lora_model`

### 非 LoRA 模式下的路径管理
- 完整模型保存在任务特定路径 `output_dir_{task_name}`
- head 模型保存在相同路径 `output_dir_{task_name}/head.bin`

### 安全的模型加载
- 自动检查模型路径是否存在，不存在则从 HuggingFace 初始化
- LoRA 模式下确保基础模型存在，不存在则保存
- 透明处理模型加载异常，提供详细日志

## 使用方法

1. 确保配置对象 `cfg` 包含必要的字段（如 `cfg.task.name`）
2. 调用 `get_base_model(cfg)` 获取模型及头部网络
3. 保存时使用模型的 `save` 方法即可，它会根据配置自动选择正确的路径

## 主要改进

1. **路径管理**：基于任务名称和 LoRA 模式自动构建路径
2. **参数过滤**：在 `encode` 方法中过滤掉不兼容的参数
3. **LoRA 任务类型**：使用 `TaskType.FEATURE_EXTRACTION` 解决与 BERT 模型的兼容性问题
4. **完善日志**：添加清晰明确的日志信息，方便调试
5. **异常处理**：添加了异常处理逻辑，提高代码健壮性



# 模型目录结构说明

## LoRA 模式目录结构

在LoRA模式下，基础模型保存在一个公共路径，而LoRA适配器和头部模型则保存在特定任务的路径中。

```
models/
├── base_model/                  # 基础模型公共路径 (cfg.model.base_backbone_path)
│   ├── model.safetensors        # 基础模型权重 (不会被更新)
│   ├── config.json              # 模型配置
│   └── ...
│
├── output_qcot                  # qcot任务的输出路径 (cfg.outputs.model_dir_qcot)
│   └── lora_head.bin                 # qcot任务的头部模型
│
├── output_qcot_lora_model       # qcot任务的LoRA适配器
│   ├── adapter_config.json      # LoRA配置
│   ├── adapter_model.bin        # LoRA权重
│   └── ...
│
├── output_semi                  # semi任务的输出路径
│   └── lora_head.bin                 # semi任务的头部模型
│
├── output_semi_lora_model       # semi任务的LoRA适配器
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
│
├── output_hard                  # hard任务的输出路径
│   └── lora_head.bin                 # hard任务的头部模型
│
└── output_hard_lora_model       # hard任务的LoRA适配器
    ├── adapter_config.json
    ├── adapter_model.bin
    └── ...
```

### LoRA模式加载流程：

1. 首先尝试从 `base_model/` 加载基础模型
2. 如果基础模型不存在，则从 HuggingFace (`base_backbone_name`) 初始化并保存
3. 根据当前任务名称，尝试加载对应的 LoRA 适配器（如 `output_qcot_lora_model/`）
4. 如果 LoRA 适配器不存在，则初始化新的 LoRA 配置
5. 尝试加载对应任务的头部模型（如 `output_qcot/head.bin`）

## 非LoRA模式目录结构

在非LoRA模式下，每个任务都有自己独立的完整模型，按照任务链（qcot -> semi -> hard）依次加载和保存。

```
models/
├── output_qcot/                 # qcot任务的输出路径
│   ├── model.safetensors        # 完整模型权重
│   ├── config.json              # 模型配置
│   ├── normal_head.bin                 # 头部模型权重
│   └── ...
│
├── output_semi/                 # semi任务的输出路径（加载自qcot输出）
│   ├── model.safetensors        # 完整模型权重
│   ├── config.json              # 模型配置
│   ├── normal_head.bin                 # 头部模型权重
│   └── ...
│
└── output_hard/                 # hard任务的输出路径（加载自semi输出）
    ├── model.safetensors        # 完整模型权重
    ├── config.json              # 模型配置
    ├── normal_head.bin                 # 头部模型权重
    └── ...
```

### 非LoRA模式加载流程：

1. 对于第一个任务（qcot）：
   - 尝试加载指定路径的模型（如果之前已经保存过）
   - 如果失败，从HuggingFace (`base_backbone_name`) 初始化

2. 对于后续任务（semi, hard）：
   - 必须从前一个任务的输出加载（qcot -> semi -> hard）
   - 如果前一个任务的输出不存在，则报错（不会尝试从HuggingFace加载）

3. 每个任务完成后，保存完整模型到自己的输出路径，供下一个任务使用

## 配置示例

```python
cfg = {
    "task": {
        "name": "semi",                      # 当前任务名称
        "chain": ["qcot", "semi", "hard"],   # 任务链定义
        "first_task_name": "qcot"            # 首个任务名称
    },
    "model": {
        "use_lora": True,                    # 是否使用LoRA
        "base_backbone_path": "models/base_model",  # 基础模型路径
        "base_backbone_name": "bert-base-uncased",  # 初始化模型名称
        "trust_remote_code": True,
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["query", "key", "value"]
        }
    },
    "outputs": {
        "model_dir": "models/output"         # 输出目录前缀
    }
}
```

