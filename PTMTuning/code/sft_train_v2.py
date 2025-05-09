"""
Chain of Thought (CoT) Distillation Training using TRL
=======================================================

This script implements supervised fine-tuning (SFT) for distilling CoT reasoning capabilities from large models to smaller ones.
Features:
- DeepSpeed integration for distributed training
- Hydra configuration management
- Mixed precision training support
- Training visualization through WandB
- Proper handling of math explanations
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/opt-350m",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/cot_data.csv",
        metadata={"help": "Path to the training data CSV file"}
    )
    fold_id: Optional[int] = field(
        default=None,
        metadata={"help": "Fold ID to use for training/validation split. If None, use all data."}
    )
    prompt_template: str = field(
        default="Q: {question}\nA: Let me think through this step by step.\n{explanation}",
        metadata={"help": "Prompt template for formatting inputs"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Percentage of data to use for validation"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for training"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./results")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    fp16: bool = field(default=True)
    report_to: str = field(default="wandb")
    run_name: str = field(default="cot-distillation")
    deepspeed: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)


class CoTDataset(Dataset):
    """Dataset for Chain-of-Thought training."""
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer, 
        prompt_template: str, 
        max_seq_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Format the prompt with CoT explanation
        prompt = self.prompt_template.format(
            question=row['QuestionText'],
            explanation=row['Explanation']
        )
        
        # Tokenize the prompt
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare the labels (same as input_ids for causal LM training)
        input_ids = tokenized["input_ids"][0]
        labels = input_ids.clone()
        
        # Set padding tokens to -100 so they're ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # For autoregressive training, we often only want to calculate loss on the explanation part
        # Find the position where the explanation starts
        question_part = self.tokenizer(
            f"Q: {row['QuestionText']}\nA: Let me think through this step by step.\n",
            return_tensors="pt"
        )["input_ids"][0]
        question_length = len(question_part)
        
        # Set labels for the question part to -100 (optional - depends on training strategy)
        # Uncomment if you want to focus loss only on explanation part
        # labels[:question_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"][0],
            "labels": labels
        }


def prepare_data(data_args):
    """Load and prepare the dataset for training and validation."""
    # Load data
    df = pd.read_csv(data_args.data_path)
    
    # Filter by fold_id if specified
    if data_args.fold_id is not None:
        train_df = df[df['fold_id'] != data_args.fold_id].reset_index(drop=True)
        val_df = df[df['fold_id'] == data_args.fold_id].reset_index(drop=True)
    else:
        # Random split
        train_size = int((1 - data_args.validation_split) * len(df))
        indices = np.random.permutation(len(df))
        train_df = df.iloc[indices[:train_size]].reset_index(drop=True)
        val_df = df.iloc[indices[train_size:]].reset_index(drop=True)
    
    logger.info(f"Training examples: {len(train_df)}")
    logger.info(f"Validation examples: {len(val_df)}")
    
    return train_df, val_df


@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """Main training function using Hydra for configuration."""
    print(OmegaConf.to_yaml(cfg))
    
    # Convert Hydra config to appropriate argument classes
    model_args = ModelArguments(**cfg.model)
    data_args = DataArguments(**cfg.data)
    training_args = TrainingArguments(**cfg.training)
    
    # Initialize wandb if enabled
    if training_args.report_to == "wandb":
        wandb.init(
            project=cfg.wandb.project,
            name=training_args.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    train_df, val_df = prepare_data(data_args)
    
    # Create datasets
    train_dataset = CoTDataset(
        train_df,
        tokenizer,
        data_args.prompt_template,
        data_args.max_seq_length
    )
    
    val_dataset = CoTDataset(
        val_df,
        tokenizer,
        data_args.prompt_template,
        data_args.max_seq_length
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field=None,  # We're handling the text formatting in our dataset
        max_seq_length=data_args.max_seq_length,
        packing=False  # We handle our own packing in the dataset
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "final_model"))
    
    # Close wandb if enabled
    if training_args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()