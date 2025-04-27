import bitsandbytes as bnb
from torch import optim


def get_optimizer_grouped_parameters(cfg, model, print_fn=print):
    param_dict = {name: param for name, param in model.named_parameters()}
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # no weight decay: usually bias or norm layers
    params_dict_no_decay = {name: param for name, param in param_dict.items() if len(param.shape) == 1}
    params_dict_decay = {name: param for name, param in param_dict.items() if len(param.shape) != 1}

    # ==== 分组逻辑 ====

    # 1. LoRA 参数（可选）
    params_dict_lora_a = {name: param for name, param in params_dict_decay.items() if "lora_A" in name}
    params_dict_lora_b = {name: param for name, param in params_dict_decay.items() if "lora_B" in name}
    params_dict_embed_tokens = {name: param for name, param in params_dict_decay.items() if "embed_tokens" in name}

    # 2. 额外 Head 层参数（比如你的自定义头部）
    head_keywords = cfg.optimizer.head_keywords if hasattr(cfg.optimizer, "head_keywords") else ["head", "classifier", "mlp", "output"]
    params_dict_head = {
        name: param for name, param in params_dict_decay.items()
        if any(keyword in name for keyword in head_keywords)
    }

    # 3. 其余参数（减去上面的所有）
    used_keys = set(params_dict_no_decay) | set(params_dict_lora_a) | set(params_dict_lora_b) | set(params_dict_head)
    params_dict_remaining = {
        name: param for name, param in params_dict_decay.items()
        if name not in used_keys
    }

    # ==== 打印信息 ====
    def print_param_group_info(group, group_name):
        n_params = round(sum(p.numel() for p in group.values()) / 1e6, 2)
        print_fn(f"{group_name}: # params: {n_params}M | Sample keys: {list(group.keys())[:2]}")

    print_param_group_info(params_dict_no_decay, "no_decay")
    if len(params_dict_lora_a) > 0: print_param_group_info(params_dict_lora_a, "lora_A")
    if len(params_dict_lora_b) > 0: print_param_group_info(params_dict_lora_b, "lora_B")
    if len(params_dict_head) > 0: print_param_group_info(params_dict_head, "custom_head")
    if len(params_dict_embed_tokens) > 0: print_param_group_info(params_dict_embed_tokens, "embed_tokens")
    print_param_group_info(params_dict_remaining, "remaining")

    # ==== 构建优化器分组 ====
    wd = cfg.optimizer.weight_decay
    lr = cfg.optimizer.lr

    optim_groups = [
        {"params": list(params_dict_no_decay.values()), "lr": lr, "weight_decay": 0.0},
    ]

    if len(params_dict_lora_a) > 0:
        optim_groups.append({
            "params": list(params_dict_lora_a.values()),
            "lr": getattr(cfg.optimizer, "lr_lora_a", lr),
            "weight_decay": wd,
        })

    if len(params_dict_lora_b) > 0:
        optim_groups.append({
            "params": list(params_dict_lora_b.values()),
            "lr": getattr(cfg.optimizer, "lr_lora_b", lr),
            "weight_decay": wd,
        })
    if len(params_dict_embed_tokens) > 0:
        optim_groups.append({
            "params": list(params_dict_embed_tokens.values()),
            "lr": getattr(cfg.optimizer, "lr_embed_tokens", lr),
            "weight_decay": wd,
        })

    if len(params_dict_head) > 0:
        optim_groups.append({
            "params": list(params_dict_head.values()),
            "lr": getattr(cfg.optimizer, "lr_head", lr),
            "weight_decay": wd,
        })

    if len(params_dict_remaining) > 0:
        optim_groups.append({
            "params": list(params_dict_remaining.values()),
            "lr": lr,
            "weight_decay": wd,
        })

    return optim_groups


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
