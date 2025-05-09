"""
基于TRL的SFT训练代码，用于CoT能力蒸馏
支持DeepSpeed、Hydra配置、混合精度训练和训练过程可视化
输入数据为 csv：fold_id,query_text,
"""

import os
import json
import logging
import random
import math
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import deepspeed
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    模型参数配置
    """
    model_name_or_path: str = field(
        metadata={"help": "训练的基础模型路径或名称"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "使用的tokenizer（如果与模型不同）"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "是否使用Flash Attention 2"}
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "是否使用PEFT/LoRA进行微调"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout率"}
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "要应用LoRA的目标模块名称列表"}
    )


@dataclass
class DataArguments:
    """
    数据参数配置
    """
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "训练数据文件路径"}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "验证数据文件路径"}
    )
    prompt_template_file: Optional[str] = field(
        default=None,
        metadata={"help": "提示模板文件路径"}
    )
    max_source_length: int = field(
        default=1024,
        metadata={"help": "输入序列的最大长度"}
    )
    max_target_length: int = field(
        default=1024,
        metadata={"help": "目标序列的最大长度"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "数据集中文本字段的名称"}
    )
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "提示字段名称"}
    )
    response_field: str = field(
        default="response",
        metadata={"help": "回复字段名称"}
    )
    options_field: Optional[str] = field(
        default="options",
        metadata={"help": "选项字段名称（如果有）"}
    )
    examples_field: Optional[str] = field(
        default="examples",
        metadata={"help": "示例字段名称（如果有）"}
    )
    pack_sequences: bool = field(
        default=True,
        metadata={"help": "是否打包序列以提高训练效率"}
    )
    add_cot_prefix: bool = field(
        default=True,
        metadata={"help": "是否在数据中添加CoT前缀以鼓励思维链输出"}
    )
    cot_prefix: str = field(
        default="Let me think step by step.\n",
        metadata={"help": "CoT前缀文本"}
    )
    random_cot_injection: float = field(
        default=0.5,
        metadata={"help": "随机注入CoT前缀的概率（0-1）"}
    )


@dataclass
class TrainingConfig:
    """
    训练配置类
    """
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)


def setup_logging(training_args):
    """
    设置日志级别
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


def load_prompt_templates(template_file):
    """
    加载提示模板
    """
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    return templates


def format_example(example, prompt_templates, data_args):
    """
    格式化单个样本，应用提示模板
    """
    template_name = example.get("template_name", "default")
    if template_name not in prompt_templates:
        template_name = "default"
    
    template = prompt_templates[template_name]
    
    # 获取基本字段
    prompt = example.get(data_args.prompt_field, "")
    response = example.get(data_args.response_field, "")
    
    # 获取可选字段
    options = example.get(data_args.options_field, None) if data_args.options_field else None
    examples = example.get(data_args.examples_field, None) if data_args.examples_field else None
    
    # 应用模板
    formatted_prompt = template
    for key, value in example.items():
        if isinstance(value, str):
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, value)
    
    # 特殊处理选项和示例
    if options and "{options}" in formatted_prompt:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        formatted_prompt = formatted_prompt.replace("{options}", options_text)
    
    if examples and "{examples}" in formatted_prompt:
        examples_text = "\n\n".join(examples)
        formatted_prompt = formatted_prompt.replace("{examples}", examples_text)
    
    # 添加CoT前缀（如果启用）
    if data_args.add_cot_prefix and random.random() < data_args.random_cot_injection:
        if not response.startswith(data_args.cot_prefix):
            response = data_args.cot_prefix + response
    
    return {
        "prompt": formatted_prompt,
        "response": response,
    }


def prepare_dataset(data_args, prompt_templates):
    """
    准备数据集
    """
    # 加载训练数据
    if data_args.train_file.endswith('.json') or data_args.train_file.endswith('.jsonl'):
        train_dataset = load_dataset('json', data_files=data_args.train_file)['train']
    elif data_args.train_file.endswith('.csv'):
        train_dataset = load_dataset('csv', data_files=data_args.train_file)['train']
    else:
        train_dataset = load_dataset(data_args.train_file)['train']
    
    # 加载验证数据（如果有）
    validation_dataset = None
    if data_args.validation_file:
        if data_args.validation_file.endswith('.json') or data_args.validation_file.endswith('.jsonl'):
            validation_dataset = load_dataset('json', data_files=data_args.validation_file)['train']
        elif data_args.validation_file.endswith('.csv'):
            validation_dataset = load_dataset('csv', data_files=data_args.validation_file)['train']
        else:
            validation_dataset = load_dataset(data_args.validation_file)['train']
    
    # 应用提示模板格式化
    def format_dataset(examples):
        formatted_examples = []
        for example in examples:
            formatted_example = format_example(example, prompt_templates, data_args)
            formatted_examples.append(formatted_example)
        return formatted_examples
    
    # 格式化数据集
    formatted_train_dataset = Dataset.from_list(format_dataset(train_dataset))
    
    if validation_dataset:
        formatted_validation_dataset = Dataset.from_list(format_dataset(validation_dataset))
    else:
        formatted_validation_dataset = None
    
    return formatted_train_dataset, formatted_validation_dataset


def prepare_model_and_tokenizer(model_args, training_args):
    """
    准备模型和分词器
    """
    # 加载tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
    
    # 加载模型
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    
    if model_args.use_flash_attention_2:
        model_kwargs["use_flash_attention_2"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # 应用PEFT/LoRA（如果启用）
    if model_args.use_peft:
        logger.info("使用PEFT/LoRA进行微调")
        if not model_args.target_modules:
            # 根据模型类型设置默认目标模块
            if "llama" in model_args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "bloom" in model_args.model_name_or_path.lower():
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            elif "gpt-neox" in model_args.model_name_or_path.lower():
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            else:
                target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            target_modules = model_args.target_modules
        
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


class LossTracker:
    """
    跟踪和可视化训练过程中的损失
    """
    def __init__(self, log_dir):
        self.train_losses = []
        self.eval_losses = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def add_train_loss(self, loss, step):
        self.train_losses.append((step, loss))
        self.writer.add_scalar("Loss/train", loss, step)
    
    def add_eval_loss(self, loss, step):
        self.eval_losses.append((step, loss))
        self.writer.add_scalar("Loss/eval", loss, step)
    
    def plot_loss(self):
        plt.figure(figsize=(12, 6))
        
        # 绘制训练损失
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            plt.plot(steps, losses, label="Training Loss", color="blue")
        
        # 绘制评估损失
        if self.eval_losses:
            steps, losses = zip(*self.eval_losses)
            plt.plot(steps, losses, label="Validation Loss", color="red")
        
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(self.log_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
        # 也保存为CSV
        df = pd.DataFrame({
            "step": [x[0] for x in self.train_losses],
            "train_loss": [x[1] for x in self.train_losses]
        })
        if self.eval_losses:
            eval_df = pd.DataFrame({
                "step": [x[0] for x in self.eval_losses],
                "eval_loss": [x[1] for x in self.eval_losses]
            })
            df = pd.merge_asof(df, eval_df, on="step")
        
        df.to_csv(os.path.join(self.log_dir, "loss_history.csv"), index=False)
        return plot_path


def get_sft_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, data_args):
    """
    创建SFTTrainer实例
    """
    max_seq_length = data_args.max_source_length + data_args.max_target_length
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=os.cpu_count(),
        packing=data_args.pack_sequences,
        formatting_func=None,  # 我们已经预先格式化了数据集
    )
    
    return trainer


class TrainCallback(transformers.TrainerCallback):
    """
    自定义回调以跟踪训练进度和损失
    """
    def __init__(self, loss_tracker):
        self.loss_tracker = loss_tracker
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if "loss" in logs:
                self.loss_tracker.add_train_loss(logs["loss"], step)
            if "eval_loss" in logs:
                self.loss_tracker.add_eval_loss(logs["eval_loss"], step)


def save_config(config, output_dir):
    """
    保存训练配置
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)


@hydra.main(config_path=None)
def main(cfg: DictConfig):
    """
    主训练函数
    """
    # 从Hydra配置创建训练参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(OmegaConf.to_container(cfg, resolve=True))
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置日志
    setup_logging(training_args)
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 保存配置
    save_config(cfg, training_args.output_dir)
    
    # 加载提示模板
    prompt_templates = {}
    if data_args.prompt_template_file:
        prompt_templates = load_prompt_templates(data_args.prompt_template_file)
    else:
        prompt_templates["default"] = "{prompt}"
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(data_args, prompt_templates)
    
    # 准备模型和分词器
    model, tokenizer = prepare_model_and_tokenizer(model_args, training_args)
    
    # 创建损失跟踪器
    loss_tracker = LossTracker(os.path.join(training_args.output_dir, "logs"))
    
    # 创建训练器
    trainer = get_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        data_args=data_args,
    )
    
    # 添加回调
    trainer.add_callback(TrainCallback(loss_tracker))
    
    # 开始训练
    train_result = trainer.train()
    
    # 保存最终模型
    trainer.save_model()
    
    # 保存训练指标
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    # 运行评估（如果有验证集）
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # 绘制损失曲线
    plot_path = loss_tracker.plot_loss()
    logger.info(f"损失曲线已保存到: {plot_path}")
    
    # 完成
    logger.info("✅ 训练完成！")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="training_config", node=TrainingConfig)
    main()