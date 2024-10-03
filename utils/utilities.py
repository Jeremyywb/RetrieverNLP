# This is the utilities.py file in utils folder

from typing import Union, Tuple, Optional, Dict, NamedTuple
import numpy as np
from enum import Enum
from pathlib import Path
from torch.utils.data import DataLoader
import time
import torch
from torch import Tensor
from collections.abc import Mapping

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
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
        passages (`np.ndarray`, *optional*): Passages data passed to the model.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        passages: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.passages = passages
        self.elements = (self.predictions, self.label_ids)
        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)
        if self.passages is not None:
            self.elements += (self.passages,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]




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


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result

def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    if not (isinstance(tensors, torch.Tensor) and isinstance(new_tensors, torch.Tensor)):
        assert (
            type(tensors) is type(new_tensors)
        ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")
    
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

class EvalLoopContainer:
    """
    Container to store intermediate results of evaluation loop

    Args:
        do_nested_concat (`bool`, *optional*, defaults to `True`):
            If set to `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.
    """

    def __init__(self, do_nested_concat: bool = True, padding_index: int = -100):
        self.do_nested_concat = do_nested_concat
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensors if self.do_nested_concat else [tensors]
        elif self.do_nested_concat:
            self.tensors = nested_concat(self.tensors, tensors, padding_index=self.padding_index)
        else:
            self.tensors.append(tensors)

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        # Check if we have something to add, if not just return
        if self.tensors is None:
            return

        new_arrays = nested_numpify(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.do_nested_concat:
            self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
        else:
            self.arrays.extend(new_arrays)

        # reset device tensors after adding to cpu
        self.tensors = None

    def get_arrays(self):
        """Returns the numpified and moved to CPU stored objects."""
        self.to_cpu_and_numpy()
        return self.arrays
    

class ProjPathsFiles:
    '''
    当前只允许在项目自目录中创建一个目录
    input :  relative path or file which in current project dir
    output : 
        if  input is none output none 
        if  input parrent dir is not in current project dir raise error
        if  input is in current project dir , fullfill absolute path of given path where ever the project root path is
    '''
    def __init__(self, relative_path_or_file):
        self.project_root = Path(__file__).resolve().parent.parent
        adust = self.check_path_in_project(relative_path_or_file)
        
        if adust is None:
            self.abs_path_or_file = None
        elif adust == False:
            raise ValueError(f"{self.__class__.__name__}:relative_path_or_file:{relative_path_or_file}'s parent path IS not path of project")
        else:
            self.abspath =  self.project_root/relative_path_or_file
            self.abs_path_or_file = str(self.abspath.as_posix())

    def check_path_in_project(self, relative_path_or_file):
        if relative_path_or_file is None:
            return None
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


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    return result