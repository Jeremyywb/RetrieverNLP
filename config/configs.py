import json
import os
from ..utils.utilities import ProjPathsFiles
from dataclasses import dataclass, field,fields
from collections import OrderedDict
from typing import Optional


def validate_fields(cls):
    original_init = cls.__init__

    def new_init(self, **kwargs):
        valid_fields = set(f.name for f in fields(cls))
        unexpected_args = set(kwargs.keys()) - valid_fields
        if unexpected_args:
            raise TypeError(f"{cls.__name__} got unexpected keyword argument(s): {', '.join(unexpected_args)}")
        original_init(self, **kwargs)

    cls.__init__ = new_init
    return cls


class BaseConfig(OrderedDict):
    """
    Base configuration class that provides common methods for managing configurations.
    """

    def __init__(self, **kwargs):
        """
        Initialize configuration from keyword arguments.
        Stores all key-value pairs as attributes of the class.
        """
        if self.__class__ is not BaseConfig and not hasattr(self.__class__, "__dataclass_fields__"):
            raise TypeError(f"Subclasses of ModelOutput must be decorated with @dataclass: {self.__class__.__name__}")
        super().__init__(**kwargs)

    def __post_init__(self):
        for field in fields(self):
            v = getattr(self, field.name)
            if v is not None:
                try:
                    self[field.name] = v
                except Exception as e:
                    print(f"{self.__class__.__name__} :Error setting {field.name} to {v}: {e}")

        if hasattr(self,'auto_fullfill_paths' ):
            for filed_name in self['auto_fullfill_paths']:
                v = getattr(self, filed_name)
                if type(v) is str: 
                    new_v = ProjPathsFiles(v).abs_path_or_file
                    setattr(self, filed_name, new_v )


    def log_config(self):
        """Return a formatted string that summarizes the configuration."""
        config_summary = f"{self.__class__.__name__} Configuration:\n"
        for key, value in self.__dict__.items():
            config_summary += f"{key}: {value}\n"
        return config_summary

    def save_to_file(self, file_path):
        """Save the configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path):
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file {file_path} does not exist.")

        with open(file_path, 'r') as f:
            config_dict = json.load(f)            
        return cls(**config_dict)
    
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    @classmethod
    def from_dict(cls, args_dict):
        return cls(**args_dict)

    def to_dict(self):
        """
        Convert the model output to a dictionary.
        """
        return dict(self)

    def validate(self):
        """Validate configuration parameters. Override in subclasses if needed."""
        raise NotImplementedError("This method should be implemented in the subclass.")


@validate_fields
@dataclass
class DataLoaderConfig(BaseConfig):
    batch_size  : int  = None
    num_workers : int  = None
    pin_memory  : int  = None
    drop_last   : bool = None
    shuffle     : bool = None

@validate_fields
@dataclass
class OptimizerConfig(BaseConfig):
    '''
    LLDR : layer wise learing rate
        if use LLDR - set LLDR between 0 and 1
        if not - set LLDR  1
    '''
    optimizer_name : str = None
    learning_rate  : float = None
    weight_decay   : float = None
    adam_beta1     : float = None
    adam_beta2     : float = None
    eps   : float = None
    LLDR  : float = None


@validate_fields
@dataclass
class SchedulerConfig(BaseConfig):
    warmup_ratio: float = None

@validate_fields
@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    last_epoch=-1

@validate_fields
@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    num_cycles: Optional[float] = 0.5
    last_epoch: Optional[int] = -1

@validate_fields
@dataclass
class CosineWithRestartsSchedulerConfig(SchedulerConfig):
    num_cycles: Optional[float] = 1.0
    last_epoch: Optional[int] = -1
@validate_fields
@dataclass
class PolynomialSchedulerConfig(SchedulerConfig):
    power: Optional[float] = 1.0
    lr_end: Optional[float] = 1e-7
    last_epoch: Optional[int] = -1

@validate_fields
@dataclass
class ConstantSchedulerConfig(SchedulerConfig):
    last_epoch: Optional[int] = -1

@validate_fields
@dataclass
class ConstantWithWarmupSchedulerConfig(SchedulerConfig):
    last_epoch = -1

@validate_fields
@dataclass
class InverseSqrtSchedulerConfig(SchedulerConfig):
    last_epoch = -1

@validate_fields
@dataclass
class ReduceOnPlateauSchedulerConfig(SchedulerConfig):
    factor: Optional[float] = 0.1
    patience: Optional[int] = 10
    threshold: Optional[float] = 1e-4
    threshold_mode: Optional[str] = 'rel'
    cooldown: Optional[int] = 0
    min_lr: Optional[float] = 0.0
    eps: Optional[float] = 1e-8


@validate_fields
@dataclass
class CosineWithMinLrSchedulerConfig(SchedulerConfig):
    num_cycles: float = 0.5
    last_epoch: int = -1
    min_lr: float = None
    min_lr_rate: float = None

@validate_fields
@dataclass
class WarmupStableDecaySchedulerConfig(SchedulerConfig):
    num_stable_steps: int = 0
    num_decay_steps: int = 0
    min_lr_ratio: float = 0
    num_cycles: float = 0.5


@validate_fields
@dataclass
class CallbackConfigs(BaseConfig):
    load_best_model_at_end  : bool = None
    logging_steps : int = None
    eval_steps : int = None
    save_steps : int = None
    early_stopping_patience  : int = None
    early_stopping_threshold : float = None
    metric_for_best_model   : str  = None
    logging_first_step      : bool = None
    greater_is_better       : bool = None
    logging_strategy : str  = None
    eval_strategy    : str  = None
    save_strategy    : str  = None
    eval_delay       : int  = None

@validate_fields
@dataclass
class RetrieverDataConfig(BaseConfig):
    passage_instruction_for_retrieval : str = None
    query_instruction_for_retrieval   : str = None
    train_group_size : int = None
    passage_max_len  : int = None
    query_max_len    : int = None
    json_path        : str = None
    csv_path         : str = None
    auto_fullfill_paths : list  = field(default_factory=lambda: ['json_path','csv_path'])

@validate_fields
@dataclass
class TrainerConfig(BaseConfig):
    gradient_accumulation_steps : int   = None
    backbone_with_params_only   : bool  = None
    torch_empty_cache_steps     : int   = None
    split_train_valid_rate      : float = None
    load_best_model_at_end      : bool  = None
    eval_do_concat_batches      : bool  = None
    passages_csv_path           : str   = None
    best_model_name             : str   = None
    num_train_epochs            : int   = None
    num_freeze_layers           : int   = None
    evaluates                   : dict  = None
    include_for_metrics         : list  = None
    max_grad_norm               : float = None
    num_devices                 : int   = None
    warmup_ratio                : float = None
    input                       : str   = None
    fp16                        : bool  = None
    task                        : str   = None
    auto_fullfill_paths         : list  = field(default_factory=lambda: ['passages_csv_path'])

@validate_fields
@dataclass
class RetrieverModelConfig(BaseConfig):
    load_from_pretrained_path   : bool = None
    load_from_finetuned_path    : bool = None
    inbatch_for_long_passage    : int  = None
    sentence_pooling_method     : str  = None
    negatives_cross_device      : bool = None
    output_hidden_states        : bool = None
    use_inbatch_neg     : bool  = None
    model_path          : str   = None
    temperature         : float = None
    model_name          : str   = None
    output_dir          : str   = None
    normlized           : bool  = None
    num_labels          : int   = None
    name                : str   = None
    auto_fullfill_paths : list  = field(default_factory=lambda: ['model_path','output_dir'])

@validate_fields
@dataclass
class TrainingConfigs(BaseConfig):
    train_dataloader : DataLoaderConfig = None
    valid_dataloader : DataLoaderConfig = None
    test_dataloader  : DataLoaderConfig = None
    callbacks        : CallbackConfigs  = None
    scheulder        : SchedulerConfig  = None
    optim            : OptimizerConfig  = None

@validate_fields   
@dataclass
class RetireverTrainingConfigs(TrainingConfigs):
    model       : RetrieverModelConfig = None
    trainer     : TrainerConfig        = None
    trainset    : RetrieverDataConfig  = None
    validset    : RetrieverDataConfig  = None
    testset     : RetrieverDataConfig  = None

    

@validate_fields
@dataclass
class RerankerModelConfig(BaseConfig):
    pass

@validate_fields
@dataclass
class RerankerDataConfig(BaseConfig):
    pass

@validate_fields
@dataclass
class RerankerTrainingConfigs(TrainingConfigs):
    model       : RerankerModelConfig = None
    trainer     : TrainerConfig        = None
    trainset    : RerankerDataConfig  = None
    validset    : RerankerDataConfig  = None
    testset     : RerankerDataConfig  = None

