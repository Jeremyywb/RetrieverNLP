from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from ..config.configs import RetrieverModelConfig
from ..models.retriever import BgeBiEncoderModel
import yaml
import torch

def load_yaml(training_config_yaml):
     with open(training_config_yaml, 'r') as file:
        config_dicts = yaml.safe_load(file)
     return config_dicts


class RetrieverInfference:
    def __init__(self, config:Union[str, dict]) -> None:
        if isinstance(config, dict):
            self.model_args = RetrieverModelConfig.from_dict(config)
        elif isinstance(config, str) :
            if config.endswith('yaml'):
                config_dict = load_yaml(config)['model']
                if 'name' in config_dict:
                    config_dict.pop('name')
                self.model_args = RetrieverModelConfig.from_dict(config_dict )
            else:
                raise ValueError(f'Not Implement Yet string {config}! only for yaml file config while is str ')
        else:
            raise ValueError('Not Implement yet! for config type only support dict and string')
        
        if not hasattr( self.model_args, 'is_infference'):
            raise ValueError("config file or dict shold provide is_infference attr")
        self.model_args['is_infference'] = True
        self.model = BgeBiEncoderModel(self.model_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, tokenized_querys):
        return self.model.encode( tokenized_querys )
    
    def encode_passages(self, tokenized_passage):
        return self.model.encode_passages(tokenized_passage)
    
    def compute_similarity(self, querys, passages, group_size):
        scores = self.model.compute_similarity(
            querys[:, None, :,], passages.view(querys.size(0), group_size, -1)
            ).squeeze(1) / self.model.temperature
        return scores.view(querys.size(0), -1)
