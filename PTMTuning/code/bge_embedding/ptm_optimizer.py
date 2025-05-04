import bitsandbytes as bnb
from torch import optim

def get_optimizer_grouped_parameters(cfg, model, print_fn=print):
    """
    为BERT+LoRA模型创建优化器参数分组，专为retrieval embedding任务优化
    
    Args:
        cfg: 配置对象，包含优化器设置
        model: 模型对象
        print_fn: 打印函数
        
    Returns:
        优化器参数分组列表
    """
    print_fn('#---IN optimizer MODEL-----------------------------------------------')
    lora_found = False
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            if param.requires_grad:
                print_fn(f"✅ Found LoRA param: {name}")
                print_fn(f"{name} requires_grad: {param.requires_grad}")
                lora_found = True
                break
    
    if not lora_found:
        print_fn("⚠️ No LoRA parameters found with requires_grad=True")
    print_fn('#---IN optimizer MODEL-----------------------------------------------')

    # 收集所有参数
    param_dict = {name: param for name, param in model.named_parameters()}
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # 设置基础学习率
    lr = getattr(cfg.optimizer, "lr", 8e-5)  # retrieval 任务默认基础学习率
    lr_lora_a = getattr(cfg.optimizer, "lr_lora_a", 1.5e-4)  # A矩阵默认学习率
    lr_lora_b = getattr(cfg.optimizer, "lr_lora_b", 6e-5)    # B矩阵默认学习率
    lr_head = getattr(cfg.optimizer, "lr_head", 4e-4)        # 头部默认学习率
    lr_embed_tokens = getattr(cfg.optimizer, "lr_embed_tokens", 5e-5)  # embedding默认学习率
    wd = getattr(cfg.optimizer, "weight_decay", 0.01)        # 权重衰减
    use_layer_lr_decay = getattr(cfg.optimizer, "use_layer_lr_decay", True)  # 是否使用层级学习率衰减
    layer_lr_decay_factor = getattr(cfg.optimizer, "layer_lr_decay_factor", 0.92)  # 层级衰减因子

    # 分离无衰减参数(通常是偏置和层归一化参数)和有衰减参数
    params_dict_no_decay = {name: param for name, param in param_dict.items() if len(param.shape) == 1}
    params_dict_decay = {name: param for name, param in param_dict.items() if len(param.shape) != 1}

    # ==== 分组逻辑 ====
    # 1. LoRA 参数
    params_dict_lora_a = {name: param for name, param in params_dict_decay.items() if "lora_A" in name}
    params_dict_lora_b = {name: param for name, param in params_dict_decay.items() if "lora_B" in name}
    
    # 2. Embedding 参数
    params_dict_embed_tokens = {name: param for name, param in params_dict_decay.items() if "embedding" in name}

    # 3. 自定义头部参数
    head_keywords = [cfg.optimizer.head_keywords] if hasattr(cfg.optimizer, "head_keywords") else ["head", "classifier", "mlp", "output"]
    params_dict_head = {
        name: param for name, param in params_dict_decay.items()
        if any(keyword in name for keyword in head_keywords)
    }

    # 4. 其余参数
    used_keys = set(params_dict_no_decay) | set(params_dict_lora_a) | set(params_dict_lora_b) | set(params_dict_head) | set(params_dict_embed_tokens)
    params_dict_remaining = {
        name: param for name, param in params_dict_decay.items()
        if name not in used_keys
    }

    # ==== 打印参数分组信息 ====
    def print_param_group_info(group, group_name):
        n_params = round(sum(p.numel() for p in group.values()) / 1e6, 2)
        print_fn(f"{group_name}: # params: {n_params}M | Sample keys: {list(group.keys())[:2]}")

    print_param_group_info(params_dict_no_decay, "no_decay")
    if len(params_dict_lora_a) > 0: print_param_group_info(params_dict_lora_a, "lora_A")
    if len(params_dict_lora_b) > 0: print_param_group_info(params_dict_lora_b, "lora_B")
    if len(params_dict_head) > 0: print_param_group_info(params_dict_head, "custom_head")
    if len(params_dict_embed_tokens) > 0: print_param_group_info(params_dict_embed_tokens, "embed_tokens")
    print_param_group_info(params_dict_remaining, "remaining")




    def get_layer_wise_lr_factors(model, base_lr_a, base_lr_b, decay_factor=0.95):
        """为不同层的LoRA参数返回学习率"""
        # 创建层级映射字典
        layer_wise_lrs = {}

        # 寻找所有LoRA参数并确定它们的层号
        for name, _ in model.named_parameters():
            if "lora" not in name:
                continue

            # 提取层号 (例如 "encoder.layer.5.attention...")
            if "encoder.layer." in name:
                layer_str = name.split("encoder.layer.")[1]
                layer_num = int(layer_str.split(".")[0])

                # 计算该层的学习率系数：较深层使用较高学习率
                lr_factor = decay_factor ** (23 - layer_num)  # 假设有24层，从0开始

                if "lora_A" in name:
                    layer_wise_lrs[name] = base_lr_a * lr_factor
                elif "lora_B" in name:
                    layer_wise_lrs[name] = base_lr_b * lr_factor

        return layer_wise_lrs

    # 添加# 添加到get_optimizer_grouped_parameters函数中
    def get_module_type_lr_factors(name, base_lr):
        """基于模块类型返回学习率调整因子"""
        if "attention.self.query" in name:
            return base_lr * 1.2  # 注意力机制的查询投影更关键
        elif "attention.self.key" in name:
            return base_lr * 0.8  # 键投影可以使用较低学习率
        elif "attention.self.value" in name:
            return base_lr * 1.0  # 值投影使用标准学习率
        elif "attention.output" in name:
            return base_lr * 0.9  # 注意力输出
        elif "intermediate" in name:
            return base_lr * 0.8  # FFN第一层
        elif "output.dense" in name and "attention" not in name:
            return base_lr * 0.7  # FFN第二层
        else:
            return base_lr  # 默认值

    # ==== Retrieval任务的特殊学习率调整 ====
    def get_retrieval_optimized_lr(name, base_lr):
        """Retrieval任务优化的学习率计算"""
        # 1. 层级衰减
        layer_lr = get_layer_wise_lr(name, base_lr)
        
        # 2. 模块类型调整
        if getattr(cfg.optimizer, "use_module_type_lr", True):
            factor = get_module_type_lr_factor(name)
            layer_lr *= factor
            
        # 3. 特殊层调整
        # 顶层特别重要
        is_top_layer = any(f"encoder.layer.{i}" in name for i in range(20, 24))
        if is_top_layer:
            layer_lr *= 1.1
            
        return layer_lr
    # 获取层级学习率调整
    if hasattr(cfg.optimizer, "use_layer_lr_decay") and cfg.optimizer.use_layer_lr_decay:
        layer_lrs = get_layer_wise_lr_factors(
            model, 
            base_lr_a=getattr(cfg.optimizer, "lr_lora_a", lr),
            base_lr_b=getattr(cfg.optimizer, "lr_lora_b", lr),
            decay_factor=getattr(cfg.optimizer, "layer_lr_decay_factor", 0.95)
        )
    else:
        layer_lrs = {}
    
    # ==== 构建优化器分组 ====
    optim_groups = [
        {"params": list(params_dict_no_decay.values()), "lr": lr, "weight_decay": 0.0, 'name': 'params_dict_no_decay'},
    ]

    if len(params_dict_lora_a) > 0:
        # 如果使用层级学习率，则为每个参数单独分组
        if layer_lrs:
            for name, param in params_dict_lora_a.items():
                param_lr = layer_lrs.get(name, getattr(cfg.optimizer, "lr_lora_a", lr))
                # 应用模块类型调整
                if hasattr(cfg.optimizer, "use_module_type_lr") and cfg.optimizer.use_module_type_lr:
                    param_lr = get_module_type_lr_factors(name, param_lr)
                
                optim_groups.append({
                    "params": [param],
                    "lr": param_lr,
                    "weight_decay": wd,
                    'name': f'lora_A_{name}'
                })
        else:
            # 否则使用统一分组
            optim_groups.append({
                "params": list(params_dict_lora_a.values()),
                "lr": getattr(cfg.optimizer, "lr_lora_a", lr),
                "weight_decay": wd,
                'name': 'params_dict_lora_a'
            })

    # 对B矩阵做类似处理...
    if len(params_dict_lora_b) > 0:
        if layer_lrs:
            for name, param in params_dict_lora_b.items():
                param_lr = layer_lrs.get(name, getattr(cfg.optimizer, "lr_lora_b", lr))
                if hasattr(cfg.optimizer, "use_module_type_lr") and cfg.optimizer.use_module_type_lr:
                    param_lr = get_module_type_lr_factors(name, param_lr)
                
                optim_groups.append({
                    "params": [param],
                    "lr": param_lr,
                    "weight_decay": wd,
                    'name': f'lora_B_{name}'
                })
        else:
            optim_groups.append({
                "params": list(params_dict_lora_b.values()),
                "lr": getattr(cfg.optimizer, "lr_lora_b", lr),
                "weight_decay": wd,
                'name': 'params_dict_lora_b'
            })
    
    # 处理embedding参数
    if len(params_dict_embed_tokens) > 0:
        optim_groups.append({
            "params": list(params_dict_embed_tokens.values()),
            "lr": lr_embed_tokens,
            "weight_decay": wd,
            'name': 'params_dict_embed_tokens'
        })

    # 处理头部参数
    if len(params_dict_head) > 0:
        optim_groups.append({
            "params": list(params_dict_head.values()),
            "lr": lr_head,
            "weight_decay": wd,
            'name': 'params_dict_head'
        })

    # 处理剩余参数
    if len(params_dict_remaining) > 0:
        optim_groups.append({
            "params": list(params_dict_remaining.values()),
            "lr": lr,
            "weight_decay": wd,
            'name': 'params_dict_remaining'
        })

    return optim_groups

# def get_optimizer_grouped_parameters(cfg, model, print_fn=print):

#     print('#---IN optimizer MODEL-----------------------------------------------')
#     for name, param in model.named_parameters():
#         if "lora" in name.lower():
#             if param.requires_grad:
#                 print("✅ Found LoRA param:", name)
#                 print(name, "requires_grad:", param.requires_grad)
#                 break
#     print('#---IN optimizer MODEL-----------------------------------------------')

        
#     param_dict = {name: param for name, param in model.named_parameters()}
#     param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

#     # no weight decay: usually bias or norm layers
#     params_dict_no_decay = {name: param for name, param in param_dict.items() if len(param.shape) == 1}
#     params_dict_decay = {name: param for name, param in param_dict.items() if len(param.shape) != 1}

#     # ==== 分组逻辑 ====

#     # 1. LoRA 参数（可选）
#     params_dict_lora_a = {name: param for name, param in params_dict_decay.items() if "lora_A" in name}
#     params_dict_lora_b = {name: param for name, param in params_dict_decay.items() if "lora_B" in name}
#     params_dict_embed_tokens = {name: param for name, param in params_dict_decay.items() if "embedding" in name}

#     # 2. 额外 Head 层参数（比如你的自定义头部）
#     head_keywords = [cfg.optimizer.head_keywords] if hasattr(cfg.optimizer, "head_keywords") else ["head", "classifier", "mlp", "output"]
#     params_dict_head = {
#         name: param for name, param in params_dict_decay.items()
#         if any(keyword in name for keyword in head_keywords)
#     }

#     # 3. 其余参数（减去上面的所有）
#     used_keys = set(params_dict_no_decay) | set(params_dict_lora_a) | set(params_dict_lora_b) | set(params_dict_head)| set(params_dict_embed_tokens)
#     params_dict_remaining = {
#         name: param for name, param in params_dict_decay.items()
#         if name not in used_keys
#     }

#     # ==== 打印信息 ====
#     def print_param_group_info(group, group_name):
#         n_params = round(sum(p.numel() for p in group.values()) / 1e6, 2)
#         print_fn(f"{group_name}: # params: {n_params}M | Sample keys: {list(group.keys())[:2]}")

#     print_param_group_info(params_dict_no_decay, "no_decay")
#     if len(params_dict_lora_a) > 0: print_param_group_info(params_dict_lora_a, "lora_A")
#     if len(params_dict_lora_b) > 0: print_param_group_info(params_dict_lora_b, "lora_B")
#     if len(params_dict_head) > 0: print_param_group_info(params_dict_head, "custom_head")
#     if len(params_dict_embed_tokens) > 0: print_param_group_info(params_dict_embed_tokens, "embed_tokens")
#     print_param_group_info(params_dict_remaining, "remaining")

#     # ==== 构建优化器分组 ====
#     wd = cfg.optimizer.weight_decay
#     lr = cfg.optimizer.lr

#     optim_groups = [
#         {"params": list(params_dict_no_decay.values()), "lr": lr, "weight_decay": 0.0,'name':'params_dict_no_decay'},
#     ]

#     if len(params_dict_lora_a) > 0:
#         optim_groups.append({
#             "params": list(params_dict_lora_a.values()),
#             "lr": getattr(cfg.optimizer, "lr_lora_a", lr),
#             "weight_decay": wd,
#             'name':'params_dict_lora_a'
#         })

#     if len(params_dict_lora_b) > 0:
#         optim_groups.append({
#             "params": list(params_dict_lora_b.values()),
#             "lr": getattr(cfg.optimizer, "lr_lora_b", lr),
#             "weight_decay": wd,
#              'name':'params_dict_lora_b'
#         })
#     if len(params_dict_embed_tokens) > 0:
#         optim_groups.append({
#             "params": list(params_dict_embed_tokens.values()),
#             "lr": getattr(cfg.optimizer, "lr_embed_tokens", lr),
#             "weight_decay": wd,
#             'name':'params_dict_embed_tokens'
#         })

#     if len(params_dict_head) > 0:
#         optim_groups.append({
#             "params": list(params_dict_head.values()),
#             "lr": getattr(cfg.optimizer, "lr_head", lr),
#             "weight_decay": wd,
#             'name':'params_dict_lr_head'
#         })

#     if len(params_dict_remaining) > 0:
#         optim_groups.append({
#             "params": list(params_dict_remaining.values()),
#             "lr": lr,
#             "weight_decay": wd,
#             'name':'params_dict_remaining'
#         })

#     return optim_groups


def get_optimizer(cfg, model, print_fn=print):
    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamW8bit": bnb.optim.Adam8bit,
    }
    assert cfg.optimizer.name in _optimizers, f"Optimizer {cfg.optimizer.name} not supported"

    optim_groups = get_optimizer_grouped_parameters(cfg, model, print_fn)

    optimizer = _optimizers[cfg.optimizer.name](
        optim_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.adam_beta_1, cfg.optimizer.adam_beta_2),
        eps=cfg.optimizer.adam_epsilon,
    )

    return optimizer
