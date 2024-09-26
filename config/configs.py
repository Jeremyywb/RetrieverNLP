import json
import os
from ..utils.utilities import ProjPathsFiles
from dataclasses import dataclass, field,fields
from collections import OrderedDict

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
                self[field.name] = v

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

    @classmethod
    def from_dict(cls, args_dict):
        return cls(**args_dict)

    def validate(self):
        """Validate configuration parameters. Override in subclasses if needed."""
        raise NotImplementedError("This method should be implemented in the subclass.")



@dataclass
class DataLoaderConfig(BaseConfig):
    batch_size  : int  = None
    num_workers : int  = None
    pin_memory  : int  = None
    drop_last   : bool = None
    shuffle     : bool = None

@dataclass
class OptimizerConfig(BaseConfig):
    learning_rate : float = None
    weight_decay  : float = None
    adam_beta1    : float = None
    adam_beta2    : float = None
    eps   : float = None
    LLDR  : float = None

@dataclass
class SchedulerConfig(BaseConfig):
    onecycle_pct_start     : float = None
    cosine_num_cycles      : float = None
    linear_last_epoch      : int   = -1
    onecycle_maxlr : float = None
    poly_lr_end    : float = None
    poly_power     : int   = 3

@dataclass
class CallbackConfigs(BaseConfig):
    load_best_model_at_end  : bool = None
    metric_for_best_model   : str  = None
    logging_first_step      : bool = None
    greater_is_better       : bool = None
    logging_strategy : str  = None
    logging_strategy : str  = None
    eval_strategy    : str  = None
    save_strategy    : str  = None
    eval_delay       : int  = None

@dataclass
class RetrieverDataConfig(BaseConfig):
    passage_instruction_for_retrieval : str = None
    query_instruction_for_retrieval   : str = None
    train_group_size : int = None
    passage_max_len  : int = None
    query_max_len    : int = None
    train_data       : str = None

@dataclass
class TrainerConfig(BaseConfig):
    pass

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
    auto_fullfill_paths : list  = field(default_factory=lambda: ['model_path','output_dir'])


@dataclass
class TrainingConfigs(BaseConfig):
    train_dataloder_cfg : DataLoaderConfig = None
    valid_dataloder_cfg : DataLoaderConfig = None
    test_dataloder_cfg  : DataLoaderConfig = None
    callbacks_cfg       : CallbackConfigs  = None
    scheulder_cfg       : SchedulerConfig  = None
    optim_cfg           : OptimizerConfig  = None
    
@dataclass
class RetireverTrainingConfigs(TrainingConfigs):
    model_cfg   : RetrieverModelConfig = None
    trainer_cfg : TrainerConfig        = None
    data_cfg    : RetrieverDataConfig  = None



