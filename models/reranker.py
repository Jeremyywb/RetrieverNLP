import torch
from pathlib import Path
import torch.distributed as dist
from torch import nn
from typing import Dict, Optional
from torch import Tensor
from transformers import AutoConfig,AutoModelForSequenceClassification
from ..utils.loggings import get_logger
from .model_output import SequenceClassifierOutput
from ..config.configs import RerankerModelConfig
import torch.nn.functional as F
import numpy as np

logger = get_logger(__name__)

class BgeCrossEncoder(nn.Module):
    def __init__(self, modelconfig : RerankerModelConfig ):
        super().__init__()

        self.load_from_pretrained_path        = modelconfig.load_from_pretrained_path
        self.load_from_finetuned_path         = modelconfig.load_from_finetuned_path
        # self.inbatch_for_long_passage         = modelconfig.inbatch_for_long_passage
        # self.sentence_pooling_method          = modelconfig.sentence_pooling_method
        # self.negatives_cross_device           = modelconfig.negatives_cross_device
        self.output_hidden_states             = modelconfig.output_hidden_states
        self.model_path           = modelconfig.model_path
        # self.use_inbatch_neg      = modelconfig.use_inbatch_neg
        # self.temperature          = modelconfig.temperature
        self.model_name           = modelconfig.model_name
        self.output_dir           = modelconfig.output_dir
        # self.normlized            = modelconfig.normlized
        self.num_labels           = modelconfig.num_labels

        self.batch_size = modelconfig.batch_size
        self.group_size = modelconfig.group_size



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

        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name_or_path , config=self.config)
        if not self.load_from_pretrained_path or not self.load_from_finetuned_path:
            self.backbone.save_pretrained(self.model_path)
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.batch_size, dtype=torch.long)
        )

    def freeze_layers(self, num_layers):
        for i in range(0,num_layers,1):
            for name, param in self.backbone.encoder.layer[i].named_parameters():
                param.requires_grad = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(self, inputs:Dict[str,Tensor]):
        return self.encode(self.batch_size, self.group_size, inputs, self.target_label)
    
    def encode(self, batch_size:int,group_size:int, inputs:Dict[str,Tensor], target_label:Tensor):
        '''
        group_size:
            train: one pos and group_size-1 of neg
            valid: retrieved doc tokenized, dose not known which one is pos or even without pos
        '''
        ranker_out = self.backbone(**inputs, return_dict=True)
        scores = ranker_out.logits.view(
            batch_size,
            group_size
        )
        loss = self.cross_entropy(scores, target_label)
        output = SequenceClassifierOutput(loss=loss, **ranker_out)
        output['logits'] = scores
        return output

    def save(self, output_dir: str):
        state_dict = self.backbone.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.backbone.save_pretrained(output_dir, state_dict=state_dict)

    def predict(self, retrieved_text_tokenized:Dict[str,Tensor], retrieved_ids:np.ndarray,device:torch.device,top_k:int):
        batch_size = len(retrieved_ids)
        retrieved_text_tokenized = {k:v.to(device) for k,v in retrieved_text_tokenized.items()}
        with torch.no_grad():
            ranker_out = self.backbone( **retrieved_text_tokenized , return_dict=True)
        logits = ranker_out.logits.view(
            batch_size,
            -1
        ).detach().cpu().numpy()
        del retrieved_text_tokenized,ranker_out
        torch.cuda.empty_cache()
        sorted_indices = np.argsort(-logits, axis=1)  # 根据第二个维度排序
        sorted_labels = retrieved_ids[np.arange(retrieved_ids.shape[0])[:, None], sorted_indices]  # 使用排序索引重新排列标签

        return sorted_labels[:,:top_k]
