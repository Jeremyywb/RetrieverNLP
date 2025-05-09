from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

from trl import SFTConfig


@dataclass
class CustomSFTConfig(SFTConfig):
    """
    扩展SFTConfig类，添加额外的配置选项
    """
    # 自定义配置
    custom_cfg: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "自定义配置，用于外部初始化"})
    
    # DeepSpeed配置
    deepspeed_config: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "DeepSpeed配置文件路径或配置字典"}
    )
    
    # LoRA配置
    lora_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "LoRA配置"}
    )
    
    # 数据集配置
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "训练数据文件路径"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件路径"}
    )
    fold_id: Optional[int] = field(
        default=0,
        metadata={"help": "使用哪个fold进行训练"}
    )
    prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "提示模板"}
    )
    
    # 优化器配置
    optimizer_type: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "优化器类型"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    
    # 学习率调度器配置
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "学习率调度器类型"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "预热比例"}
    )
    warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "预热步数"}
    )