import torch
from torch import nn
from typing import Dict
from ..utils.loggings import get_logger




class BaseModelForFinetune(nn.Module):
    def __init__(self, **kwargs):
        pass

    def forward(self, batch_input: Dict[str, torch.Tensor] ):
        pass

    def free_top_n_layers(self, module, n):
        raise NotImplementedError
    
    def compute_loss(self, logit, label_id ):
        raise NotImplementedError
    

class BaseRetrieverModelFFT(BaseModelForFinetune):
    def __init__(self, **kwargs ):
        pass
    
    def forward(self, batch_input: Dict[str, torch.Tensor]):
        pass
    
    def compute_loss(self, logit, label_id):
        pass
    
    def free_top_n_layers(self, module, n):
        raise NotImplementedError
    

    
