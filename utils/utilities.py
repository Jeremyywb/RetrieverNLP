# This is the utilities.py file in utils folder

from typing import Union, Tuple, Optional, Dict, NamedTuple
import numpy as np
from enum import Enum
from pathlib import Path
from torch.utils.data import DataLoader

class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(ExplicitEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"

def has_length(dataset):
    try:
        return len(dataset) is not None
    except TypeError:
        return False



class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


class ProjPathsFiles:
    '''当前只允许在项目自目录中创建一个目录'''
    def __init__(self, relative_path_or_file):
        self.project_root = Path(__file__).resolve().parent.parent
        if not self.check_path_in_project(relative_path_or_file):
            raise ValueError(f"relative_path_or_file:{relative_path_or_file}'s parent path IS not path of project")
        self.abspath =  self.project_root/relative_path_or_file
        self.abs_path_or_file = str(self.abspath.as_posix())

    def check_path_in_project(self, relative_path_or_file):
        top_level_dirs = [p.name for p in self.project_root.iterdir() if p.is_dir()]
        input_path = Path(relative_path_or_file)
        path_parts = input_path.parts
        # 如果是文件（即路径有文件名和上级目录）
        if input_path.suffix:  # 通过判断是否有后缀，判断是否是文件
            # 获取文件的上一级目录名称
            parent_dir = path_parts[0] if len(path_parts) > 1 else None
            if parent_dir in top_level_dirs:
                return True
            else:
                return False
        
        else:  # 如果是目录
            # 获取目录的一级目录名称
            if len(relative_path_or_file.split("/"))>2:
                return False
            top_level_dir = path_parts[0]  # 获取路径的第一个部分
            if top_level_dir in top_level_dirs:
                return True
            else:
                return False
