from dataclasses import dataclass
from typing import Optional
from torch import Tensor,nn
from transformers.modeling_outputs import ModelOutput
# from transformers import BitsAndBytesConfig, AutoModel
# from transformers import AutoConfig

from modelscope import BitsAndBytesConfig, AutoModel
from modelscope import AutoConfig

from peft import LoraConfig, TaskType, get_peft_model
import torch
from pathlib import Path
from  typing import Dict
from metrics.metrics import UnifiedCoTLoss
import torch.nn.functional as F

from peft import PeftModel

from pathlib import Path
import os
from omegaconf import OmegaConf, DictConfig

@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    c_reps: Optional[Tensor] = None
    r_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None





def initialize_model_paths(cfg):
    """
    Initialize and return all model paths based on configuration.
    
    Creates a standardized set of paths for model components that can be
    shared across different modules (model loading, tokenizer, etc.)
    
    Compatible with Hydra configuration objects.
    
    Args:
        cfg: Hydra configuration object with model settings
        
    Returns:
        dict: Dictionary containing all relevant paths and path-related information
    """
    # Extract task info from config
    

    task_name = cfg.task.name if hasattr(cfg, 'task') and hasattr(cfg.task, 'name') else "default"
    base_backbone_path = cfg.model.base_backbone_path  # Common path for base model in LoRA mode
    base_backbone_name = cfg.model.base_backbone_name  # HF model name for initialization
    
    # Get task chain information
    task_chain = cfg.task.chain if hasattr(cfg.task, 'chain') else ["qcot", "semi", "hard"]
    if task_name in task_chain and task_chain.index(task_name) > 0:
        prev_task_idx = task_chain.index(task_name) - 1
        prev_task_name = task_chain[prev_task_idx]
    else:
        prev_task_name = None
    
    # Define first task name with fallback
    first_task_name = cfg.task.first_task_name if hasattr(cfg.task, 'first_task_name') else "qcot"
    
    # Set appropriate head model binary name based on LoRA usage
    head_bin_name = 'lora_head' if cfg.model.use_lora else 'normal_head'
    
    # Define paths based on LoRA usage
    if cfg.model.use_lora:
        # In LoRA mode, backbone is common, but task-specific adapters
        backbone_path = base_backbone_path
        task_specific_path = f"{cfg.outputs.model_dir}_{prev_task_name}" if prev_task_name else None  
    else:
        backbone_path = f"{cfg.outputs.model_dir}_{prev_task_name}" if prev_task_name else base_backbone_name
        task_specific_path = backbone_path

    lora_adapter_path =  f"{task_specific_path}_lora_model" if (cfg.model.use_lora and task_specific_path is not None) else None
    save_task_specific_path = f"{cfg.outputs.model_dir}_{task_name}"
    save_lora_adapter_path = f"{save_task_specific_path}_lora_model"

    # Check for existing checkpoints
    is_backbone_exists = Path(backbone_path).exists() and (Path(backbone_path) / "model.safetensors").exists()
    is_lora_adapter_exists = lora_adapter_path and Path(lora_adapter_path).exists()
    if task_specific_path is None:
        is_task_checkpoint_exists = False
    else:
        is_task_checkpoint_exists = Path(task_specific_path).exists() and (Path(task_specific_path) / f"{head_bin_name}.bin").exists()
    
    
    # Create tokenizer path - typically same as backbone model path
    tokenizer_path = backbone_path if is_backbone_exists else base_backbone_name

    tokenizer_save_path = base_backbone_path if cfg.model.use_lora else save_task_specific_path

    # Construct dictionary of all paths and related information
    paths_info = {
        # Core paths
        "backbone_path": backbone_path,
        "base_backbone_path": base_backbone_path,
        "base_backbone_name": base_backbone_name,
        "task_specific_path": task_specific_path,
        "lora_adapter_path": lora_adapter_path,
        "tokenizer_path": tokenizer_path,
        "tokenizer_save_path":tokenizer_save_path,
        'save_task_specific_path':save_task_specific_path,
        'save_lora_adapter_path':save_lora_adapter_path,
        # Task information
        "task_name": task_name,
        "prev_task_name": prev_task_name,
        "first_task_name": first_task_name,
        "task_chain": task_chain,
        
        # Model component names
        "head_bin_name": head_bin_name,
        
        # Existence flags
        "is_backbone_exists": is_backbone_exists,
        "is_lora_adapter_exists": is_lora_adapter_exists,
        "is_task_checkpoint_exists": is_task_checkpoint_exists
    }
    
    return paths_info


def add_paths_to_config(cfg):
    """
    Compute all model paths and add them to the configuration object.
    
    This function enhances the Hydra config object by adding a new 'paths' section
    containing all model-related paths information.
    
    Args:
        cfg: Hydra configuration object with model settings
        
    Returns:
        cfg: Updated configuration object with paths information
    """
    # Get paths dictionary
    OmegaConf.set_struct(cfg, False)
    paths_info = initialize_model_paths(cfg)
    
    # With Hydra, we can directly add a new node to the configuration tree
    # Convert the paths dictionary to an OmegaConf object
    paths_conf = OmegaConf.create(paths_info)
    
    # Add paths to config
    # This will work if cfg is a DictConfig, which is the case with Hydra configs
    if not hasattr(cfg, "paths"):
        # Merge the new paths into the existing config
        cfg = OmegaConf.merge(cfg, {"paths": paths_conf})
    else:
        # If paths already exists, update it
        cfg.paths = paths_conf
    
    return cfg



def get_base_model(cfg):
    """
    Load base model with or without LoRA based on configuration.
    
    For LoRA mode:
    - Base backbone model is loaded from/saved to a common path
    - LoRA adapters and head model are stored in task-specific paths
    
    For non-LoRA mode:
    - Complete model follows task chain (qcot -> semi -> hard)
    - Each task reads from previous task's output path
    - First task can initialize from base model if needed
    
    Args:
        cfg: Configuration object with model settings
        
    Returns:
        tuple: (base_model, head_model)
    """
    # Extract paths and task info from config
        # Get paths from config - cfg.paths should be initialized already
    backbone_path = cfg.paths.backbone_path
    base_backbone_name = cfg.paths.base_backbone_name
    task_specific_path = cfg.paths.task_specific_path
    lora_adapter_path = cfg.paths.lora_adapter_path
    head_bin_name = cfg.paths.head_bin_name
    
    # Get existence flags
    is_backbone_exists = cfg.paths.is_backbone_exists
    is_lora_adapter_exists = cfg.paths.is_lora_adapter_exists
    is_task_checkpoint_exists = cfg.paths.is_task_checkpoint_exists


    # Load configuration from appropriate source
    if is_backbone_exists:
        config = AutoConfig.from_pretrained(
            backbone_path,
            trust_remote_code=cfg.model.trust_remote_code
        )
    else:
        # For non-LoRA with missing previous task checkpoint, this will be caught later
        config = AutoConfig.from_pretrained(
            base_backbone_name,
            trust_remote_code=cfg.model.trust_remote_code
        )
    config.use_cache = False
    
    # Set up model parameters
    torch_dtype = torch.bfloat16
    model_kwargs = {
        "config": config,
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation
    }
    
    # Add quantization if needed
    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    
    # Load or initialize base model
    if cfg.model.use_lora:
        # LoRA mode - can fall back to initial model if needed
        try:
            # Try to load existing base model
            base_model = AutoModel.from_pretrained(backbone_path, **model_kwargs)
            print(f"✅ Loaded base model from: {backbone_path}")
        except Exception as e:
            # If loading fails, initialize from HF and save base model
            print(f"⚠️ Failed to load model from {backbone_path}: {e}")
            print(f"Initializing model from {base_backbone_name}...")
            base_model = AutoModel.from_pretrained(base_backbone_name, **model_kwargs)
            
            # Save base model to common path
            print(f"Saving base model to {backbone_path}")
            os.makedirs(Path(backbone_path).parent, exist_ok=True)
            base_model.save_pretrained(backbone_path, safe_serialization=True)
    else:
        # Non-LoRA mode - must load from previous task's output
        # First task (e.g. qcot) can load from base_backbone_name if backbone_path doesn't exist
        if cfg.task.name == cfg.task.first_task_name if hasattr(cfg.task, 'first_task_name') else "qcot":
            base_model = AutoModel.from_pretrained(base_backbone_name, **model_kwargs)
        else:
            # Subsequent tasks must load from previous task's output
            if not is_backbone_exists:
                raise ValueError(
                    f"❌ ERROR: Cannot load model for task '{cfg.task.name}'. "
                    f"Previous task checkpoint not found at {backbone_path}. "
                    f"Please ensure the previous task completed successfully."
                )
            base_model = AutoModel.from_pretrained(backbone_path, **model_kwargs)
            print(f"✅ Loaded model from previous task: {backbone_path}")
    
    # Apply LoRA if configured
    if cfg.model.use_lora:
        if is_lora_adapter_exists:
            # Load existing LoRA adapter
            print(f"✅ Loading LoRA adapter from: {lora_adapter_path}")
            base_model = PeftModel.from_pretrained(
                base_model, 
                lora_adapter_path,
                is_trainable=True  # ⭐ 关键参数 ⭐
            )
            print('#---IN GET MODEL-----------------------------------------------')
            for name, param in base_model.named_parameters():
                if "lora" in name.lower():
                    if param.requires_grad:
                        print("✅ Found LoRA param:", name)
                        print(name, "requires_grad:", param.requires_grad)
                        break
            print('#---IN GET MODEL-----------------------------------------------')
            
        else:
            # Initialize new LoRA adapter
            print(f"⚠️ No existing LoRA adapter found. Initializing new adapter.")
            target_modules = list(cfg.model.lora.target_modules) if hasattr(cfg.model.lora, 'target_modules') else []
            modules_to_save = list(cfg.model.lora.modules_to_save) if hasattr(cfg.model.lora, 'modules_to_save') else []
            
            peft_config = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.lora_alpha,
                lora_dropout=cfg.model.lora.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,  # Appropriate for BERT models
                inference_mode=False,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )
            base_model = get_peft_model(base_model, peft_config)
            # for name, param in base_model.named_parameters():
            #     if "lora" in name.lower():
            #         print("✅ Found LoRA param:", name)
            #         print(name, "requires_grad:", param.requires_grad)
            #     else:
            #         print("❌ Not LoRA param:", name)

    
    # Initialize or load head model
    head_model = SharedAlignment(hidden_size=config.hidden_size, torch_dtype=torch_dtype)
    to_load = Path(task_specific_path) / f"{head_bin_name}.bin" if cfg.task.name != 'qcot' else 'NO NEED FOR qcot stage' 
    # Load head model if task checkpoint exists
    if is_task_checkpoint_exists:
        print(f"✅ Loading head model from: {task_specific_path}")
        head_state_dict = torch.load(to_load , map_location="cpu")
        head_model.load_state_dict(head_state_dict)
    else:
        print(f"⚠️ No existing head model found:{to_load}. Using newly initialized head.")

    return base_model, head_model
        
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

    def save(self, cfg: str):
        # if isinstance(self.backbone,PeftModel):
        #     self.backbone.save_pretrained(f'{output_dir}_lora_model')
        # elif (Path(output_dir) / "model.safetensors").exists():
        #     self.backbone.save_pretrained(output_dir, safe_serialization=True)
            
        # output_path = Path(output_dir) / "head.bin"
        # torch.save(self.headmodel.state_dict(), output_path)
        """
        Save model based on LoRA or standard configuration.
        
        For LoRA mode:
        - LoRA adapters are saved to task-specific path with _lora_model suffix
        - Head model is saved to task-specific path
        
        For non-LoRA mode:
        - Full model is saved to task-specific path
        """
        # Save model based on type
        if isinstance(self.backbone, PeftModel):
            # LoRA mode - save adapter to task-specific path
            self.backbone.save_pretrained(cfg.paths.save_lora_adapter_path)
            print(f"✅ Saved LoRA adapter to: {cfg.paths.save_lora_adapter_path}")
        else:
            # Non-LoRA mode - save full model to task-specific path
            self.backbone.save_pretrained(cfg.paths.save_task_specific_path, safe_serialization=True)
            print(f"✅ Saved full model to: {cfg.paths.save_task_specific_path}")
      
        
        
        # Save head model to task-specific path
        head_path = f"{cfg.paths.save_task_specific_path}/{cfg.paths.head_bin_name}.bin"
        torch.save(self.headmodel.state_dict(), head_path)
        print(f"✅ Saved head model to: {head_path}")