import torch
from pathlib import Path
import torch.distributed as dist
from torch import nn
from typing import Dict, Optional
from torch import Tensor
from transformers import AutoModel,AutoConfig
from ..utils.loggings import get_logger
from .model_output import BgeRetrieverModelOuput
from ..config.configs import RetrieverModelConfig
import torch.nn.functional as F

logger = get_logger(__name__)

class BgeBiEncoderModel(nn.Module):
    def __init__(self, modelconfig : RetrieverModelConfig ):
        super().__init__()

        self.load_from_pretrained_path        = modelconfig.load_from_pretrained_path
        self.load_from_finetuned_path         = modelconfig.load_from_finetuned_path
        self.inbatch_for_long_passage         = modelconfig.inbatch_for_long_passage
        self.sentence_pooling_method          = modelconfig.sentence_pooling_method
        self.negatives_cross_device           = modelconfig.negatives_cross_device
        self.output_hidden_states             = modelconfig.output_hidden_states
        self.model_path           = modelconfig.model_path
        self.use_inbatch_neg      = modelconfig.use_inbatch_neg
        self.temperature          = modelconfig.temperature
        self.model_name           = modelconfig.model_name
        self.output_dir           = modelconfig.output_dir
        self.normlized            = modelconfig.normlized
        self.num_labels           = modelconfig.num_labels

        model_name_or_path = None
        if self.load_from_pretrained_path: #output_dir
             self.config = AutoConfig.from_pretrained(self.model_path ,
                                                      output_hidden_states=self.output_hidden_states, 
                                                      num_labels = self.num_labels
                                                )
             model_name_or_path = self.model_path
        elif self.load_from_finetuned_path:
            self.config = AutoConfig.from_pretrained( self.output_dir ,
                            output_hidden_states    = self.output_hidden_states, 
                            num_labels              = self.num_labels
                         )
            model_name_or_path = self.output_dir
            
        else:
            self.config = AutoConfig.from_pretrained(self.model_name ,
                            output_hidden_states =   self.output_hidden_states, 
                            num_labels           =   self.num_labels
                        )
            self.config.save_pretrained(self.model_path)#output_dir
            model_name_or_path = self.model_name


        for attr in ['hidden_dropout','hidden_dropout_prob',
                    'attention_dropout','attention_probs_dropout_prob']:
            if hasattr(self.config,attr):
                 setattr(self.config,attr,0.0)

        self.config.save_pretrained(self.output_dir)
        self.config.save_pretrained(self.model_path)

        self.backbone = AutoModel.from_pretrained(model_name_or_path , config=self.config)
        if (not self.load_from_pretrained_path) or (not self.load_from_finetuned_path):
            self.backbone.save_pretrained(self.model_path)
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if not self.normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        if self.normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def freeze_layers(self, num_layers):
        for i in range(0,num_layers,1):
            for name, param in self.backbone.encoder.layer[i].named_parameters():
                param.requires_grad = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.backbone(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()
    
    def encode_passages(self, passages):
        # new implement
        total_passages = passages['input_ids'].size(0)
        if total_passages <= self.inbatch_for_long_passage:
            return self.encode(passages)
        else:
            p_reps_list = []
            for i in range(0, total_passages, self.inbatch_for_long_passage):
                p_rep = self.encode({key: val[i:i + self.inbatch_for_long_passage] for key, val in passages.items()})
                p_reps_list.append( p_rep )
                del p_rep
            p_reps = torch.cat( p_reps_list, axis=0 )
            del p_reps_list
            return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passages: Dict[str, Tensor] = None, teacher_score: Tensor = None):

        q_reps = self.encode(query)
        p_reps = self.encode_passages(passages)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                # forward 对于一个query，内部分批次 ，每批次分配一个pos
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        del p_reps
        return BgeRetrieverModelOuput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=None,
        )

    # def compute_loss(self, scores, target):
    #     return self.cross_entropy(scores, target)
    def compute_loss(self,scores, target, margin=1.0):
        """
        计算 Pairwise Ranking Loss (Hinge Loss)
        
        参数:
        scores: (batch_size, num_doc) - 每个查询的候选文档相似度分数
        target: (batch_size, 1) - 正样本在每个查询中的索引
        margin: float - 正样本与负样本的分数差的最小阈值
        
        返回:
        loss: (batch_size,) - 每个查询的损失值
        """
        batch_size, num_doc = scores.shape
        
        # 去掉 target 的维度，变成 (batch_size,) 
        target = target
        
        # 提取正样本的分数 (batch_size,)
        positive_scores = scores.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 创建负样本掩码 (正样本位置为 False，负样本位置为 True)
        negative_mask = torch.ones_like(scores, dtype=torch.bool)
        negative_mask.scatter_(1, target.unsqueeze(1), False)
        
        # 提取所有负样本的分数 (batch_size, num_doc-1)
        negative_scores = scores.masked_select(negative_mask).view(batch_size, -1)
        
        # 计算最大负样本分数 (hardest negative) (batch_size,)
        hardest_negative_scores = negative_scores.max(dim=1).values
        
        # 计算 Hinge Loss
        loss = F.relu(margin - (positive_scores - hardest_negative_scores))
        
        # 返回损失
        return loss.mean()


    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.backbone.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.backbone.save_pretrained(output_dir, state_dict=state_dict)