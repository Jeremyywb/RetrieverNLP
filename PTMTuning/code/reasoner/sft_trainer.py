from typing import Dict, List, Optional, Union, Any
import torch
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled
from trl import SFTTrainer
import warnings
from reasoner.sft_optimizer import get_optimizer
from utils.train_utils import get_custom_cosine_schedule_with_warmup

class CustomSFTTrainer(SFTTrainer):
    """
    继承SFTTrainer并自定义优化器和学习率调度器创建方法，以适应DeepSpeed配置
    """
    
    def __init__(self, cfg=None, *args, **kwargs):
        self.cfg = cfg
        super().__init__(*args, **kwargs)
    
    def create_optimizer(self):
        """
        创建自定义优化器，支持deepspeed集成
        
        Returns:
            优化器实例
        """
        # 如果启用了deepspeed，并且optimizers已经在deepspeed中初始化，则直接返回
        if self.deepspeed:
            if getattr(self.deepspeed, "optimizers", None) is not None:
                return self.deepspeed.optimizers
        
        # 获取优化器参数
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        return get_optimizer(self.cfg, opt_model)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_custom_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler