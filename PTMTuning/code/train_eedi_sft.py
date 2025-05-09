import os
import hydra
from omegaconf import DictConfig
import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig

# CUDA backend optimizations
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# Allow TF32 on Ampere and later GPUs for matmul and cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from reasoner.sft_dataset import MathDataset
from reasoner.sft_trainer import CustomSFTTrainer
from reasoner.sft_loader import TextCollator
from utils.train_utils import train_valid_split

def plot_loss(losses, output_dir):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

@hydra.main(config_path="../conf/qwen_reasoner", config_name="conf_reasoner_7b")
def main(cfg: DictConfig):
    # 1. 数据加载与预处理
    # 假设 cfg.data.dataset_name 对应本地或远端 pandas 文件
    df = pd.read_csv(cfg.dataset.query_dataset)
    train_df, valid_df = train_valid_split(cfg, df)
    
    math_ds = MathDataset(cfg)
    tokenized_train_ds = math_ds.get_dataset(train_df)
    tokenized_valid_ds = math_ds.get_dataset(valid_df)

    # 2. 分词器 & 模型准备
    tokenizer = math_ds.tokenizer

    data_collator = TextCollator(tokenizer=tokenizer, pad_to_multiple_of=16)
    def create_custom_model(cfg):
        """ 自定义模型初始化函数，包含所有你的配置条件 """
        # BNB 量化配置
        bnb_config = None
        if cfg.model.use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head"],
            )

        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            quantization_config=bnb_config if cfg.model.use_bnb else None,
            torch_dtype=torch.bfloat16 if not cfg.model.use_bnb else None,
            attn_implementation="flash_attention_2" if not cfg.model.use_bnb else None,
        )
        
        # 通用配置
        model.config.pretraining_tp = 1
        model.config.use_cache = False
        
        # 梯度检查点
        if cfg.model.use_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # LoRA 配置（若启用）
        if cfg.model.use_lora:
            peft_config = LoraConfig(
                use_dora=cfg.model.lora.use_dora,
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.lora_alpha,
                lora_dropout=cfg.model.lora.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=cfg.model.lora.target_modules,
                modules_to_save=cfg.model.lora.modules_to_save,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model
    model = create_custom_model(cfg)
    # 3. SFTConfig 参数配置
    training_args = SFTConfig(
        output_dir=cfg.outputs.model_dir,
        num_train_epochs=cfg.train_params.num_train_epochs,
        per_device_train_batch_size=cfg.train_params.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train_params.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
        learning_rate=cfg.optimizer.lr,
        logging_steps=cfg.train_params.eval_frequency,
        bf16=True,  # 使用 bf16 代替 fp16
        fp16=False,  # 禁用 fp16
        optim="adamw_torch",
        warmup_ratio=cfg.train_params.warmup_pct,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        deepspeed=cfg.get("train", {}).get("deepspeed", None),
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=cfg.train_params.eval_frequency,
        save_steps=cfg.train_params.eval_frequency,
    )

    # 4. 初始化 CustomSFTTrainer
    trainer = CustomSFTTrainer(
        cfg=cfg,
        model=model, 
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 5. 开始训练
    result = trainer.train()
    losses = result.training_loss if hasattr(result, "training_loss") else []

    # 6. 画 loss 曲线
    if losses:
        plot_loss(losses, os.getcwd())

    # 7. 保存模型
    if cfg.save_model:
        trainer.save_model(cfg.outputs.model_dir)

if __name__ == '__main__':
    main()