import logging
import math
import os
import shutil
import uuid
from collections import defaultdict

import datasets
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin




logger = get_logger(__name__)


def generate_random_string():
    return str(uuid.uuid4())


def print_line(logger=None):
    prefix, unit, suffix = "#", "~~", "#"
    if logger is None:
        print(prefix + unit * 50 + suffix)
    else:
        logger.print(prefix + unit * 50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm%ds" % (m, s)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"] * 1e6


class AverageMeter(object):
    """Computes and stores the average and current value using exponential smoothing.
    Maintains similar structure to the original version with additional smoothing functionality.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Smoothing factor, adjustable according to the desired responsiveness
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # Initialized as 0; assumes first update sets to first value directly if required
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == n:  # First update
            self.avg = val
        else:
            self.avg = self.alpha * val + (1 - self.alpha) * self.avg  # Apply exponential smoothing


def save_checkpoint(cfg, state, is_best):
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    name = "ptm"

    filename = f"{cfg.outputs.model_dir}/{name}_last.pth"
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, f"{cfg.outputs.model_dir}/{name}_best.pth")



# ---------------------------EMA
# ### **一、核心作用**
# - **平滑模型参数更新**：通过维护模型参数的移动平均值，减少训练过程中参数剧烈波动带来的噪声。
# - **提升模型泛化性**：最终使用EMA后的参数（更稳定版本）进行推理，常用于验证/测试阶段。
# - **防御过拟合**：EMA参数相比原始参数对噪声更鲁棒，尤其在数据量少时效果显著。


# ### **三、典型使用场景**
# #### **1. 训练流程示例**
# ```python
# ema = EMA(model, decay=0.999)
# ema.register()  # 初始化影子参数

# for batch in dataloader:
#     loss = model(batch)
#     loss.backward()
#     optimizer.step()
#     ema.update()  # 更新EMA参数

# # 验证阶段
# ema.apply_shadow()  # 使用EMA参数
# evaluate(model)
# ema.restore()  # 恢复原始参数继续训练
# ```


class EMA:
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def enable_cuda_optimizations():
    # TF32 仅对 Ampere 及更新架构 GPU（如 A100、RTX 3090+）有效
    # 1.2~2倍加速 < 混合精度方法
    # TF32精度接近FP32，但计算速度更快
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
    torch.backends.cudnn.allow_tf32 = True


def setup_training_run(cfg):
    """set up training run

    Args:
        cfg: config for the training run
    """

    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
            log_with="wandb",
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.wandb.run_name}},
        )
    elif cfg.use_deepspeed_plugin:
        ds_config = {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "min_loss_scale": 1,
            "hysteresis": 2
        },
        "zero_optimization": {
            "stage": 2,  # 升级至 Stage 2
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "allgather_bucket_size": 5e7,
            "reduce_bucket_size": 5e7,
            "contiguous_gradients": True
            },
            "gradient_accumulation_steps": cfg.train_params.grad_accumulation_steps,
            "train_micro_batch_size_per_gpu": cfg.train_params.retriever_bs,
            "gradient_clipping": cfg.optimizer.max_grad_norm,
            "steps_per_print": 50
        }
        # ds_plugin = DeepSpeedPlugin(
        #     zero_stage=2,
        #     gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
        #     offload_optimizer_device="none",
        #     offload_param_device="none",
        #     fp16=True,  # 显式启用 FP16
        #     fp16_opt_level="O2",  # 优化级别
        #     gradient_clipping=0.5,  # 梯度裁剪阈值
        # )
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
        
        accelerator = Accelerator(
            deepspeed_plugin=ds_plugin,
            mixed_precision="fp16",  # 显式启用混合精度
            gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps
        )
        # 创建 DeepSpeed 插件
        # ds_plugin = DeepSpeedPlugin(
        #     zero_stage=1,
        #     gradient_accumulation_steps = cfg.train_params.grad_accumulation_steps,
        #     offload_optimizer_device="none",  # 可选，如果想使用 CPU 卸载
        #     offload_param_device="none"
        # )
        
        # # 使用插件初始化 Accelerator
        # accelerator = Accelerator(gradient_accumulation_steps=None, deepspeed_plugin=ds_plugin)
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
             mixed_precision="fp16"
        )
    accelerator.print(f"using wandb: {cfg.use_wandb}")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    # if accelerator.is_main_process:
    #     os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    if cfg.enable_cuda_optimizations:
        enable_cuda_optimizations()
    return accelerator


# --------------------scheduler----------------------------

def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 10% of it,
    following a cosine curve, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of times the learning rate will decay to 10% of the maximum learning rate. Default: 0.5 (half a cycle).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale to decay to 10% of the max lr
        decay_target = 0.1  # Decay to 10% of the max lr
        decay_factor = (1 - decay_target) * cosine_decay + decay_target

        return decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# def get_cosine_schedule_with_warmup_and_minlr(
#     optimizer,
#     num_warmup_steps,
#     num_training_steps,
#     min_lr=1e-7,
#     num_cycles=0.5,
#     last_epoch=-1
#     ):
#     # 获取初始学习率
#     base_lrs = [group['lr'] for group in optimizer.param_groups]
    
#     def lr_lambda(current_step, base_lr):
#         # Warmup phase
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
        
#         # Progress after warmup
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
#         # Cosine decay with specified number of cycles
#         cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
#         # 计算当前学习率
#         lr = cosine_decay * base_lr
        
#         # 确保不低于最小学习率
#         return max(min_lr, lr) / base_lr  # 返回相对于 base_lr 的乘数
    
#     # 为每个参数组创建一个lambda函数
#     lr_lambdas = [lambda step, base_lr=base_lr: lr_lambda(step, base_lr) for base_lr in base_lrs]
    
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch)


def get_cosine_schedule_with_warmup_and_minlr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr=1e-7,
    num_cycles=0.5,
    last_epoch=-1
):
    """
    Create a schedule with a learning rate that:
    1. Increases linearly during warmup from 0 to the initial lr set in optimizer
    2. Follows a cosine curve with the specified number of cycles
    3. Never falls below the specified min_lr
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        min_lr: Minimum learning rate to use (absolute value, not a ratio)
        num_cycles: Number of cosine cycles to complete (default: 0.5)
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        A PyTorch learning rate scheduler
    """
    # 获取初始学习率
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def lr_lambda(current_step, base_lr):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Cosine decay with specified number of cycles
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Ensure learning rate doesn't fall below min_lr
        return max(cosine_decay, min_lr / base_lr)
    
    # 为每个参数组创建一个lambda函数
    lr_lambdas = [lambda step, base_lr=base_lr: lr_lambda(step, base_lr) for base_lr in base_lrs]
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch)


def log_gradient_norms(accelerator, model, step):
    grad_l2_norm = 0.0
    param_logs = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = torch.norm(param.grad, 2).item()
            # param_logs[f"{name}_grad_l2_norm"] = norm
            grad_l2_norm += norm

    # Log aggregate norms and individual parameter norms
    accelerator.log(
        {
            "step": step,
            "total_grad_l2_norm": grad_l2_norm,
            **param_logs,  # Unpack and log individual parameter norms
        }
    )


def is_nan(x):
    return x != x


def train_valid_split(cfg, df):
    fold_df = pd.read_csv(cfg.dataset.folder_dataset)
    df = pd.merge(df, fold_df, on="query_id", how="left")
    df["fold_id"] = df["fold_id"].fillna(99).astype(int)
    print(f"# of folds: {df['fold_id'].nunique()}")
    print("Fold distribution:")
    print(df["fold_id"].value_counts())

    if cfg.full_fit:
        train_df = df.copy()
    else:
        train_df = df[df["fold_id"] != cfg.fold].copy()
    valid_df = df[df["fold_id"] == cfg.fold].copy()

    train_df = train_df.drop(columns=["fold_id"]).reset_index(drop=True)
    valid_df = valid_df.drop(columns=["fold_id"]).reset_index(drop=True)

    print(f"# of train: {train_df.shape[0]}")
    print(f"# of valid: {valid_df.shape[0]}")

    return train_df, valid_df


import gzip
import json
import pandas as pd

def load_ext_cot(file_path: str) -> pd.DataFrame:
    """
    从 .json.gz 文件中读取每行一个 JSON 对象，转换为 pandas DataFrame。

    参数:
        file_path: 压缩 JSON 文件路径

    返回:
        pandas DataFrame
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError:
                    continue  # 跳过格式错误的行
    df = pd.DataFrame(data)
    df = df.rename(columns = {'explanation':'Explanation'})
    return df