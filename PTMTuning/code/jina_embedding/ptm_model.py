from dataclasses import dataclass
from typing import Optional
from torch import Tensor,nn
from transformers.modeling_outputs import ModelOutput
from transformers import BitsAndBytesConfig, AutoModel
from transformers import AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
from pathlib import Path
from  typing import Dict
from ..metrics import UnifiedCoTLoss
import torch.nn.functional as F


@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    c_reps: Optional[Tensor] = None
    r_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None



def get_base_model(cfg):
    backbone_path = cfg.model.backbone_path
    is_local_checkpoint = Path(backbone_path).exists() and (Path(backbone_path) / "pytorch_model.bin").exists()
    from modelscope.models import Model
    config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=cfg.model.trust_remote_code)
    config.use_cache = False
    torch_dtype = torch.bfloat16
    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        base_model = AutoModel.from_pretrained(
            backbone_path,
            config=config,
            trust_remote_code=cfg.model.trust_remote_code,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation
        )
    else:
        base_model = AutoModel.from_pretrained(
            backbone_path,
            config=config,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=cfg.model.attn_implementation
        )

    # 如果有 LoRA 配置，加载 LoRA
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )
        base_model = get_peft_model(base_model, peft_config)

    # 加入 head 网络
    head_model = SharedAlignment(hidden_size=config.hidden_size,torch_dtype=torch_dtype)

    # ⚠️ 如果是读取前一阶段的 checkpoint，并且包含了 head，就要加载 state_dict
    if is_local_checkpoint:
        print(f"✅ 加载前一阶段head模型参数：{backbone_path}")
        state_dict = torch.load(Path(backbone_path) / "head.bin", map_location="cpu")
        head_model.load_state_dict(state_dict)
        state_dict = torch.load(Path(backbone_path) / "pytorch_model.bin", map_location="cpu")
        base_model.load_state_dict(state_dict, strict=False)

    return base_model, head_model


# class SharedAlignment(nn.Module):
#     def __init__(self, hidden_size: int=768, torch_dtype: Optional[torch.dtype] = None):
#         super().__init__()
#         # 共享底层变换
#         self.base_proj = nn.Linear(hidden_size, hidden_size*2)
#         self.act = nn.GELU()
        
#         # 任务特定变换
#         self.q_final = nn.Linear(hidden_size*2, hidden_size)
#         self.cot_final = nn.Linear(hidden_size*2, hidden_size)
#         self.r_final = nn.Linear(hidden_size*2, hidden_size)
#         if torch_dtype is not None:
#             # 这一步会把所有权重和 buffer 一次性转成你要的 dtype
#             self.to(torch_dtype)

#     def forward(self, x, mode):
#         shared = self.act(self.base_proj(x))
#         if mode == 'q':
#             out = self.q_final(shared)
#         elif mode == 'cot':
#             out = self.cot_final(shared)
#         else:
#             out = self.r_final(shared)
#         return F.normalize(out, p=2, dim=-1) 


class SharedAlignment(nn.Module):
    def __init__(self, hidden_size: int=768, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        # 共享底层变换
        self.base_proj = nn.Linear(hidden_size, hidden_size*2)
        self.act = nn.GELU()
        
        # 任务特定变换
        self.q_final = nn.Linear(hidden_size*2, hidden_size)
        self.cot_final = nn.Linear(hidden_size*2, hidden_size)
        self.r_final = nn.Linear(hidden_size*2, hidden_size)
        if torch_dtype is not None:
            # 这一步会把所有权重和 buffer 一次性转成你要的 dtype
            self.to(torch_dtype)

        # —— 关键：零初始化，让这条支路初始输出恒等于 0 ——，避免初始权重偏差过大
        for layer in (self.base_proj, self.q_final, self.cot_final, self.r_final):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, mode):
        # 先算出“delta”，然后 + x——这样初始时 delta≡0，输出 x
        delta = self.act(self.base_proj(x))
        if mode == 'q':
            delta = self.q_final(delta)
        elif mode == 'cot':
            delta = self.cot_final(delta)
        else:
            delta = self.r_final(delta)
        out = x + delta
        # return F.normalize(out, p=2, dim=-1)
        return F.normalize(out, p=2, dim=-1)
    


        

class BgeBiEncoderModel(nn.Module):
    def __init__(self, cfg, model, headmodel, accelerator=None):
        super().__init__()

        self.backbone = model
        self.headmodel = headmodel
        self.config = self.backbone.config
        self.sub_batch_size  = cfg.train_params.sub_batch_size
        self.loss_fun = UnifiedCoTLoss(
            alpha=cfg.train_params.loss.alpha, beta=cfg.train_params.loss.beta, gamma=cfg.train_params.loss.gamma)
        self.sentence_pooling_method = cfg.model.sentence_pooling_method
        
        # -----------------------accelerator distributed training-----------------------
        self.accelerator = accelerator
        self.negatives_cross_device = cfg.model.negatives_cross_device
        if self.negatives_cross_device:
            assert accelerator.use_distributed, "Distributed training is required for negatives_cross_device"
        self.world_size = accelerator.num_processes
        self.process_rank = accelerator.process_index

        accelerator.print(f"negatives_cross_device: {self.negatives_cross_device}")
        accelerator.print(f"world_size: {self.world_size}")
        accelerator.print(f"process_rank: {self.process_rank}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "last":
            return self.last_token_pool(hidden_state, mask)


    def encode(self, features, mode):
        if self.sub_batch_size is not None and self.sub_batch_size > 0:
            all_p_reps = []
            for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                #memory usage care
                end_inx = min(i + self.sub_batch_size, len(features["attention_mask"]))
                sub_features = {k: v[i:end_inx] for k, v in features.items()}
                last_hidden_state = self.backbone(**sub_features, return_dict=True).last_hidden_state
                p_reps = self.sentence_embedding(last_hidden_state, sub_features["attention_mask"])
                all_p_reps.append(p_reps)
                del p_reps, last_hidden_state, sub_features
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
        else:
            last_hidden_state = self.backbone(**features, return_dict=True).last_hidden_state
            all_p_reps = self.sentence_embedding(last_hidden_state, features["attention_mask"])

        return self.headmodel(all_p_reps, mode)
    def forward(self, 
                query: Dict[str, Tensor] = None, 
                cot: Dict[str, Tensor] = None, 
                contents: Dict[str, Tensor] = None,):
        q_reps = self.encode(query, mode='q')
        c_reps = self.encode(cot, mode='cot')
        r_reps = self.encode(contents, mode='r')
        loss = self.compute_loss(q_reps, c_reps, contents = r_reps)

        return EmbedderOutput(
            q_reps=q_reps,
            c_reps=c_reps,
            r_reps=r_reps,
            loss=loss
        )
        


    def compute_loss(self, 
                     anchor: torch.Tensor, 
                     cot: torch.Tensor, 
                     contents: torch.Tensor=None
                    ):
        return self.loss_fun(anchor, cot, contents = contents)

    def save(self, output_dir: str):
        state_dict = self.backbone.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.backbone.save_pretrained(output_dir, state_dict=state_dict)
        output_path = Path(output_dir) / "head.bin"
        torch.save(self.headmodel.state_dict(), output_path)

