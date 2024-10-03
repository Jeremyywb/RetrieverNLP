import os
import yaml
import torch
import json
import random
import math
import time
import shutil
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_scheduler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from ..data.collator import BgeEmbedCollator
from transformers    import DataCollatorWithPadding,PreTrainedTokenizer
from ..utils.utilities import EvalPrediction
from ..models.model_output import TrainOutput
from ..training.callbacks import (
    TrainerCallback,TrainerState,TrainerControl,DefaultFlowCallback,PrinterCallback ,
    EarlyStoppingCallback,ProgressCallback,ExportableState,CallbackHandler)
from ..config.typemapping import ConfigType,ModelType,DatasetType,ScheType,TrainerConfigType
from ..utils.loggings import get_logger
from ..data.dataset import BgeRetrieverDataset
from ..models.retriever import BgeBiEncoderModel
from ..utils.utilities import speed_metrics
from ..config.configs import (
        LinearSchedulerConfig
        ,CosineSchedulerConfig
        ,CosineWithRestartsSchedulerConfig
        ,PolynomialSchedulerConfig
        ,ConstantSchedulerConfig
        ,ConstantWithWarmupSchedulerConfig
        ,InverseSqrtSchedulerConfig
        ,ReduceOnPlateauSchedulerConfig
        ,CosineWithMinLrSchedulerConfig
        ,WarmupStableDecaySchedulerConfig

        ,RetrieverDataConfig
        ,DataLoaderConfig
        ,RetrieverModelConfig
        ,CallbackConfigs
        ,OptimizerConfig
        ,TrainerConfig
        ,RetireverTrainingConfigs
        ,RerankerTrainingConfigs
        ,RerankerModelConfig
        ,RerankerDataConfig

)
from collections.abc import Mapping

from ..utils.utilities import EvalLoopContainer
import numpy as np
from ..data.dataset import BgeRetrieverEvalDataset
from ..utils.utilities import EvalLoopOutput
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from packaging import version
parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")

TYPE_TO_DATASET_CLS = {
        DatasetType.BGERETRIEVERTRAIN:BgeRetrieverDataset,
        DatasetType.BGERETRIEVERVALID:BgeRetrieverEvalDataset,
        DatasetType.BGERETRIEVERTEST:BgeRetrieverEvalDataset,

}

TYPE_TO_CONFIG_CLS = {
        ConfigType.TRAINSET:RetrieverDataConfig,
        ConfigType.VALIDSET:RetrieverDataConfig,
        ConfigType.TESTSET:RetrieverDataConfig,
        ConfigType.BGERETRIEVERTRAIN:RetrieverDataConfig,
        ConfigType.BGERETRIEVERVALID:RetrieverDataConfig,
        ConfigType.BGERETRIEVERTEST:RetrieverDataConfig,
        ConfigType.TRAINLOADER:DataLoaderConfig,
        ConfigType.VALIDLOADER:DataLoaderConfig,
        ConfigType.TESTLOADER:DataLoaderConfig,
        ConfigType.BGEEMBEDDING:RetrieverModelConfig,
        ConfigType.CALLBACKS:CallbackConfigs,
        ConfigType.SCHEDULER:None,
        ConfigType.OPTIMIZER:OptimizerConfig,
        ConfigType.TRAINER:TrainerConfig
}


TYPE_TO_SCHEDULER_CFG_CLS = {
        ScheType.LINEAR: LinearSchedulerConfig,
        ScheType.COSINE: CosineSchedulerConfig,
        ScheType.COSINE_WITH_RESTARTS: CosineWithRestartsSchedulerConfig,
        ScheType.POLYNOMIAL: PolynomialSchedulerConfig,
        ScheType.CONSTANT: ConstantSchedulerConfig,
        ScheType.CONSTANT_WITH_WARMUP: ConstantWithWarmupSchedulerConfig,
        ScheType.INVERSE_SQRT: InverseSqrtSchedulerConfig,
        ScheType.REDUCE_ON_PLATEAU: ReduceOnPlateauSchedulerConfig,
        ScheType.COSINE_WITH_MIN_LR: CosineWithMinLrSchedulerConfig,
        # ScheType.WARMUP_STABLE_DECAY: WarmupStableDecaySchedulerConfig
}
TYPE_TO_MODEL_CLS = {
        ModelType.BGEEMBEDDING:BgeBiEncoderModel,

}

TYPE_TO_TRAINER_CLS = {
        TrainerConfigType.BGERETRIEVERTRAINER:RetireverTrainingConfigs,
        TrainerConfigType.RERANKERTRAINER:RerankerTrainingConfigs,
}


def load_yaml(training_config_yaml):
     with open(training_config_yaml, 'r') as file:
        config_dicts = yaml.safe_load(file)
     return config_dicts


# task == retrieval

logger = get_logger(__name__)
class Trainer:
    def __init__(
                self,
                training_config_yaml_or_dict: Union[str, dict] = None,
                model_name_or_instance: Union[str,nn.Module] = None,
                args: RetireverTrainingConfigs = None,
                data_collator: Optional[Union[BgeEmbedCollator,DataCollatorWithPadding]] = None,
                train_dataset: Optional[Dataset] = None,
                eval_dataset: Optional[Dataset] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                callbacks: Optional[List[TrainerCallback]] = None,
                optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None ):

        
        #  配置初始化
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        if args is None:
            if isinstance(training_config_yaml_or_dict, str):
                config_dicts = load_yaml(training_config_yaml_or_dict)
                self.args = self.set_all_config(config_dicts)
            else:
                self.args = self.set_all_config( training_config_yaml_or_dict )
        else:
            self.args = args
        print(self.args)

        self.accelerator = None
        self.is_deepspeed_enabled = False

        if not os.path.exists(self.args.model.model_path):
            os.makedirs(self.args.model.model_path)
        if not os.path.exists(self.args.model.output_dir):
            os.makedirs(self.args.model.output_dir)
        # if self.args.callbacks.best_model_checkpoint is not None:
        #     if not os.path.exists(self.args.callbacks.best_model_checkpoint):
        #         os.makedirs(self.args.callbacks.best_model_checkpoint)
        if self.args.trainer.num_devices >1:
            raise ValueError("num_devices > 1 is not supported yet")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if tokenizer is None:
            if self.args.model.load_from_pretrained_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model.model_path)
                self.tokenizer.save_pretrained(self.args.model.model_path)
            elif self.args.model.load_from_finetuned_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model.output_dir)
                self.tokenizer.save_pretrained(self.args.model.output_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model.model_name)
                if self.args.model.output_dir is None or self.args.model.model_path is None:
                    raise ValueError("model_path and output_dir should be provided for saving tokenizer")
                self.tokenizer.save_pretrained(self.args.model.output_dir)
                self.tokenizer.save_pretrained(self.args.model.model_path)
            
        else:
            self.tokenizer = tokenizer
            print(f"tokenizer is provided by user. and not saved!!!")

        if self.args.trainer.input is not None:
            train_file,valid_file,file_type = self.create_train_vaild_file(self.args.trainer.input,self.args.trainer.split_train_valid_rate)
            if file_type == 'json':
                setattr(self.args.trainset,'json_path',train_file)
                setattr(self.args.validset,'json_path',valid_file)
            elif file_type == 'csv':
                setattr(self.args.trainset,'csv_path',train_file)
                setattr(self.args.validset,'csv_path',valid_file)
            else:
                raise ValueError("Unsupported file type")
            
        if train_dataset is not None and self.args.trainer.input is not None:
            raise ValueError("train_dataset and input cannot be both provided")
        if eval_dataset is not None and self.args.trainer.input is not None:
            raise ValueError("eval_dataset and input cannot be both provided")

        if train_dataset is None:
            self.train_dataset = self.get_train_dataset()
        else:
            self.train_dataset = train_dataset

        if eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset()
        else:
            self.eval_dataset = eval_dataset
        if data_collator is None:
            raise ValueError("data_collator is required")
        

        self.data_collator = data_collator

        if isinstance(model_name_or_instance, str):
            name = ModelType(self.args.model.model_type)
            _cls = TYPE_TO_MODEL_CLS[name]
            self.model = _cls(self.args.model)
        else:
            self.model = model_name_or_instance

        if self.args.trainer.num_freeze_layers is not None:
            self.model.freeze_layers(self.args.model.num_freeze_layers)
        self.model.to(self.device)

        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()

        # 早停若是epoch，则总step 按每个epoch的step单独计算 accumulate step 再累计
        # self.warmup_ratio = self.args.scheulder.warmup_ratio
        # total_steps_without_accumulation = self.args.trainer.num_train_epochs * len(self.train_dataloader)
        # self.accumulation_steps = self.args.trainer.gradient_accumulation_steps
        # self.num_training_steps = total_steps_without_accumulation // self.accumulation_steps

        #暂时实现 按照step
        self.warmup_ratio = self.args.scheulder.warmup_ratio
        total_steps_without_accumulation = self.args.trainer.num_train_epochs * len(self.train_dataloader)
        self.accumulation_steps = self.args.trainer.gradient_accumulation_steps
        self.num_training_steps = total_steps_without_accumulation // self.accumulation_steps
        if total_steps_without_accumulation % self.accumulation_steps != 0:
            self.num_training_steps += 1
        self.num_warmup_steps = int(self.num_training_steps * self.warmup_ratio)



        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.get_optimizer_and_scheduler()
        
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.args.callbacks.early_stopping_patience,
            early_stopping_threshold=self.args.callbacks.early_stopping_threshold
        )
        default_callbacks = [DefaultFlowCallback,PrinterCallback, early_stopping_callback ]
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.state.is_local_process_zero,
            is_world_process_zero=self.state.is_world_process_zero,
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )


        # use_amp  not implemented yet

        self.current_flos = 0
        self.control = self.callback_handler.on_init_end(self.args.callbacks, self.state, self.control)

        self._train_batch_size = self.args.train_dataloader.batch_size

    def train(self):
        self.callback_handler.on_train_begin(self.args.callbacks, self.state, self.control)
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self.state.is_hyper_param_search = False
        self.state.train_batch_size = self.args.train_dataloader.batch_size

        if self.args.callbacks.logging_steps is not None:
            if self.args.callbacks.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.num_training_steps * self.args.callbacks.logging_steps)
            else:
                self.state.logging_steps = self.args.callbacks.logging_steps
      

        if self.args.callbacks.eval_steps is not None:
            if self.args.callbacks.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.num_training_steps * self.args.callbacks.eval_steps)
            else:
                self.state.eval_steps = self.args.callbacks.eval_steps

        if self.args.callbacks.save_steps is not None:
            if self.args.callbacks.save_steps < 1:
                self.state.save_steps = math.ceil(self.num_training_steps * self.args.callbacks.save_steps)
            else:
                self.state.save_steps = self.args.callbacks.save_steps
        num_examples = len(self.train_dataset)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {self.args.trainer.num_train_epochs:,}")
        logger.info(f"  Gradient Accumulation steps = {self.args.trainer.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.num_training_steps:,}")
        self.state.epoch = 0
        self.state.global_step = 0
        self.state.max_steps = self.num_training_steps
        self.state.num_train_epochs = self.args.trainer.num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True
        start_time = time.time()

        tr_loss = torch.tensor(0.0).to(self.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.model.zero_grad()
        self.optimizer.zero_grad()

        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(self.args.callbacks, self.state, self.control)
        total_batched_samples = 0
        for epoch in range(self.args.trainer.num_train_epochs):
            steps_in_epoch = len(self.train_dataloader)
            self.train_dataloader.set_epoch(epoch)
            for step, batch in enumerate(self.train_dataloader):
                total_batched_samples += self.args.train_dataloader.batch_size
                
                tr_loss_step = self.training_step(batch, step)
                if tr_loss.device != tr_loss_step.device:
                    raise ValueError("tr_loss and tr_loss_step should be on the same device")
                tr_loss += tr_loss_step
                # if (self.state.global_step+1) % self.args.callbacks.logging_steps == 0:
                #     self.log_metrics()

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch<= self.accumulation_steps and (step+1) == steps_in_epoch
                )
                if (
                    (self.state.global_step+1) % self.accumulation_steps == 0 or
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    if self.args.trainer.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.trainer.max_grad_norm)
                    
                    self.control = self.callback_handler.on_pre_optimizer_step(self.args.callbacks, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args.callbacks, self.state, self.control)
                    
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step+1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args.callbacks, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, epoch, self.state, self.control)
                else:
                    self.control = self.callback_handler.on_substep_end(self.args.callbacks, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args.callbacks, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, epoch, self.state, self.control)

        logger.info("Training completed. do not forget to save the model")
        if self.args.trainer.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        self.callback_handler.on_train_end(self.args.callbacks, self.state, self.control)

        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step
        metrics = speed_metrics(split='train', start_time=start_time, num_samples=total_batched_samples, num_steps=self.state.global_step)
        metrics['train_loss'] = train_loss
        self._save_model(self.args.model.output_dir, is_stop_training=True)

        return TrainOutput(
            self.state.global_step,
            self._total_loss_scalar / self.state.global_step,
            metrics
        )
                
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)

    def training_step(self,batch,step):
        self.model.train()
        if self.args.trainer.task == 'retrieval':
            batch.pop('passag_id')
        inputs = self._prepare_input(batch)
        outputs = self.model(**inputs)
        loss = outputs.loss
        loss = loss / self.args.trainer.gradient_accumulation_steps
        loss.backward()
        if self.args.trainer.torch_empty_cache_steps is not None:
            if (self.state.global_step+1) % self.args.trainer.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()
        return loss

    def _maybe_log_save_evaluate(self,tr_loss, grad_norm):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs : Dict[str,float] = {}
            _train_loss = tr_loss.item() 
            tr_loss -= tr_loss  # 清空tr_loss 
            logs['loss'] = round(_train_loss/(self.state.global_step - self._globalstep_last_logged),4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            self._total_loss_scalar += _train_loss
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)
        metrics = None
        if self.control.should_evaluate:
            evaluate_datasets = self.args.trainer.evaluates
            evaluate_dataloader = {}
            if evaluate_datasets['evaluate_on_train']:
                evaluate_dataloader['train'] = self.get_train_dataloader()
            if evaluate_datasets['evaluate_on_valid']:
                evaluate_dataloader['eval'] = self.get_eval_dataloader()
            # if evaluate_datasets['evaluate_on_test']:
            #     evaluate_dataloader['test'] = self.get_eval_dataloader()
            metrics = self.evaluate(eval_name='datasets',evaluate_dataloader=evaluate_dataloader)
        
        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args.callbacks, self.state, self.control)
    
    def evaluate(self,eval_name:str,evaluate_dataloader:Union[Dict[str,DataLoader],DataLoader]):
        '''recursively call evaluate if given dict of dataset'''
        # evaluate_loop
        if isinstance(evaluate_dataloader,dict):
            metrics = {}
            for eval_dateset_name,dataloader in evaluate_dataloader.items():
                dataset_metrics = self.evaluate(eval_dateset_name, dataloader)
                metrics.update(dataset_metrics)
            return metrics
        output = self.evaluate_loop(
            eval_name,
            evaluate_dataloader,
            description=f"Evaluation on {eval_name}"
        )
        self.control = self.callback_handler.on_evaluate(self.args.callbacks, self.state, self.control, output.metrics)
        self.log(output.metrics)
        return output.metrics
    
    def predict_passages(self,passages):
        self.model.eval()
        passages_inputs = self.train_dataset.prepare_tokens(self.tokenizer, passages, self.args.trainset.passage_max_len)
        passages_inputs = {k:v.to(self.args.device) for k,v in passages_inputs.items()}
        _max_length = passages_inputs['attention_mask'].sum(dim=1).max().item()
        for k,v in passages_inputs.items():
            passages_inputs[k] = v[:,:_max_length]
        outputs = self.model.encode_passages(passages_inputs)
        del passages_inputs
        return outputs


    def evaluate_loop(self,eval_name,evaluate_dataloader,description):
        psg_reps = None
        if self.args.trainer.task == 'retrieval':
            passages = self.load_passages()
            psg_reps = self.predict_passages(passages)
        self.model.eval()
        self.callback_handler.eval_dataloader = evaluate_dataloader

        all_losses = EvalLoopContainer(do_nested_concat=self.args.trainer.eval_do_concat_batches)
        all_preds = EvalLoopContainer(do_nested_concat=self.args.trainer.eval_do_concat_batches)
        all_labels = EvalLoopContainer(do_nested_concat=self.args.trainer.eval_do_concat_batches)
        all_inputs = EvalLoopContainer(do_nested_concat=self.args.trainer.eval_do_concat_batches)

        metircs = None

        eval_set_kwargs = {}
        observed_num_samples = 0
        num_samples = len(evaluate_dataloader)
        for step,inputs in enumerate(evaluate_dataloader):
            losses, logits, labels = self.prediction_step(self.model, inputs)
            if losses is not None:
                all_losses.add(losses)
            if logits is not None:
                all_preds.add(logits)
                observed_num_samples += len(all_preds)
            if labels is not None:
                all_labels.add(labels)
            if inputs is not None:
                all_inputs.add(inputs)

            self.control = self.callback_handler.on_prediction_step(self.args.callbacks, self.state, self.control)
            all_losses.to_cpu_and_numpy()
            all_preds.to_cpu_and_numpy()
            all_labels.to_cpu_and_numpy()
            all_inputs.to_cpu_and_numpy()
            del losses, logits, labels, inputs
            torch.cuda.empty_cache()
            
        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        if psg_reps is not None:
            psg_reps = psg_reps.detach().cpu().numpy()

        if (
            (self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None) or 
            (
                self.args.trainer.task == 'retrieval'
                and all_preds is not None
            )
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in self.args.trainer.include_for_metrics else None
            eval_set_kwargs["passages"] = psg_reps if "passages" in self.args.trainer.include_for_metrics else None
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids= all_labels ,**eval_set_kwargs))

            self.log(metrics)
        #map@25

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{eval_name}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{eval_name}_loss"] = all_losses.mean().item()
        # if hasattr(self, "jit_compilation_time"):
        #     metrics[f"{eval_name}_jit_compilation_time"] = self.jit_compilation_time
        # if hasattr(self, "model_preparation_time"):
        #     metrics[f"{eval_name}_model_preparation_time"] = self.model_preparation_time

        for key in list(metrics.keys()):
            if not key.startswith(f"{eval_name}_"):
                metrics[f"{eval_name}_{key}"] = metrics.pop(key)
        #####到这里
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(self,model,inputs):
        model.eval()
        if self.args.trainer.task == 'retrieval':
            labels = inputs.pop('passag_id')
        inputs = self._prepare_input(inputs)
        outputs = self.model(**inputs)
        logits = outputs.q_reps
        loss = outputs.loss
        return loss, logits, labels

    def load_passages(self):
        # passages file with passages_id and passages
        passages_csv_path = self.args.trainer.passages_csv_path
        passages_df = pd.read_csv(passages_csv_path)
        return passages_df.passages.tolist()
    


    def _save_checkpoint(self, metrics):
        output_dir = self.args.model.output_dir
        self.best_model_latest  = (Path(output_dir)/'best_model_checkpoint').as_posix()
        self.best_model_previous  =  (Path(output_dir)/'best_model_checkpoint_prev').as_posix()
        

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.callbacks.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value

                if os.path.exists(self.best_model_latest):
                    shutil.rmtree(self.best_model_previous)
                if os.path.exists(self.best_model_latest):
                    shutil.move(self.best_model_latest, self.best_model_previous)
                    os.mkdir(self.best_model_latest)
                
                self._save_model(self.best_model_latest)
                self.state.best_model_checkpoint = self.best_model_latest   

    def _save_model(self,output_dir, is_stop_training=False):
        if is_stop_training:
            self.copy_contents( self.state.best_model_checkpoint, self.args.model.output_dir )
        else:
            if self.args.trainer.backbone_with_params_only:
                self.model.backbone.save_pretrained(output_dir)
            else:
                state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, self.args.trainer.best_model_name))


    def copy_contents(self,src_dir, dest_dir):
        # 遍历源目录中的所有文件和文件夹
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dest_item = os.path.join(dest_dir, item)

            # 如果是文件夹，递归进入并复制内容
            if os.path.isdir(src_item):
                if not os.path.exists(dest_item):
                    os.makedirs(dest_item)
                # 递归复制子目录中的内容
                self.copy_contents(src_item, dest_item)
            # 如果是文件，直接使用 shutil.copy2 覆盖拷贝
            else:
                shutil.copy2(src_item, dest_item)

    def _load_best_model(self):
        pass



    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def get_optimizer_and_scheduler(self):
        print(self.args.optim)
        if self.args.trainer.num_freeze_layers is not None:
            layers = list(self.model.backbone.encoder.layer[ self.args.trainer.num_freeze_layers:])
        else:
            layers = [self.model.backbone.embeddings] + list(self.model.backbone.encoder.layer)
        layers.reverse()
        grouped_parameters = []
        _LR = self.args.optim.learning_rate
        _WD = self.args.optim.weight_decay
        LLDR = self.args.optim.LLDR
        no_decay = ['bias','LayerNorm.weight','LayerNorm.bias']
        for i,layer in enumerate(layers):
            if LLDR is not None:
                layer_lr = _LR * (LLDR ** i)
            else:
                layer_lr = _LR
            grouped_parameters.extend([
                {
                    'params':[p for n,p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    'lr':layer_lr,
                    'weight_decay':_WD
                },
                {
                    'params':[p for n,p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    'lr':layer_lr,
                    'weight_decay':0.0
                }
            ])
        optimizer_name = self.args.optim.optimizer_name
        optimizer_params = {
            'lr':_LR,
            'weight_decay':_WD,
            'betas':(self.args.optim.adam_beta1,self.args.optim.adam_beta2),
            'eps':self.args.optim.eps
        }
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(grouped_parameters,**optimizer_params)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(grouped_parameters,**optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer_name} is not supported")
        
        lr_scheduler_kwargs = self.args.scheulder.to_dict()
        lr_scheduler_kwargs.update({'num_warmup_steps':self.num_warmup_steps,'num_training_steps':self.num_training_steps})
        if self.lr_scheduler is None:
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                scheduler_specific_kwargs= lr_scheduler_kwargs
            )
        return optimizer,lr_scheduler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is required")

        dataloader_params = {
            "batch_size":self.args.train_dataloader.batch_size,
            "shuffle":self.args.train_dataloader.shuffle,
            "num_workers":self.args.train_dataloader.num_workers,
            "pin_memory":self.args.train_dataloader.pin_memory,
            "drop_last":self.args.train_dataloader.drop_last,
            "collate_fn":self.data_collator
        }
        return DataLoader(self.train_dataset,**dataloader_params)
        
    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            raise ValueError("eval_dataset is required")
        
        # if self.args.trainer.task == 'retrieval': 
        #     dataloader_params = {
        #         "batch_size":self.args.valid_dataloader.batch_size,
        #         "shuffle":self.args.valid_dataloader.shuffle,
        #         "num_workers":self.args.valid_dataloader.num_workers,
        #         "pin_memory":self.args.valid_dataloader.pin_memory,
        #         "drop_last":self.args.valid_dataloader.drop_last,
        #     }
        # else:
        dataloader_params = {
            "batch_size":self.args.valid_dataloader.batch_size,
            "shuffle":self.args.valid_dataloader.shuffle,
            "num_workers":self.args.valid_dataloader.num_workers,
            "pin_memory":self.args.valid_dataloader.pin_memory,
            "drop_last":self.args.valid_dataloader.drop_last,
            "collate_fn":self.data_collator 
        }
        
        return DataLoader(self.eval_dataset,**dataloader_params)
        
    def create_train_vaild_file(self,input_path,split_train_valid_rate):
        '''
        input_path: json# csv is not implemented yet
        split_train_valid_rate: float
        return: train_file,valid_file,file_type
        '''
        data = self.load_json_file(input_path)
        random.shuffle(data)
        train_data = data[:int(len(data)*split_train_valid_rate)]
        valid_data = data[int(len(data)*split_train_valid_rate):]
        if input_path.endswith('.json'):
            train_file = input_path.replace('.json', '_train_fold0.json')
            valid_file = input_path.replace('.json', '_valid_fold0.json')
            file_type = 'json'
        elif input_path.endswith('.csv'):
            train_file = input_path.replace('.csv', '_train_fold0.csv')
            valid_file = input_path.replace('.csv', '_valid_fold0.csv')
            file_type = 'csv'
        else:
            raise ValueError("Unsupported file type")
        self.save_json_file(train_data,train_file)
        self.save_json_file(valid_data,valid_file)
        return train_file,valid_file,file_type

    def load_json_file(self,input_path):
        data = []
        with open(input_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def save_json_file(self,data,output_path):
        with open(output_path, 'w') as file:
            for line in data:
                file.write(json.dumps(line) + '\n') # 写入文件 

    def get_train_dataset(self):
        name = DatasetType(self.args.trainset.dataset_type)
        _cls = TYPE_TO_DATASET_CLS[name]
        if self.args.trainset.json_path is not None:
            return _cls.load_from_json(self.args.trainset,self.tokenizer)
        elif self.args.trainset.csv_path is not None:
            return _cls.load_from_csv(self.args.trainset,self.tokenizer)
        else:
            raise ValueError("No input file provided")

    def get_eval_dataset(self):
        name = DatasetType(self.args.validset.dataset_type)
        _cls = TYPE_TO_DATASET_CLS[name]
        if self.args.validset.json_path is not None:
            return _cls.load_from_json(self.args.validset,self.tokenizer)
        elif self.args.validset.csv_path is not None:
            return _cls.load_from_csv(self.args.validset,self.tokenizer)
        else:
            raise ValueError("No input file provided")
    
    def set_all_config(self, config_dicts):
        cls_dict = {}
        trainerType = None
        for key in config_dicts:
            print(key)
            if key == 'name':
                continue
            value = config_dicts[key]
            if key == 'trainer':
                trainerType = value.pop('trainerType')
            if key == ConfigType.SCHEDULER:
                cls_dict[key] = self.get_scheduler_config(value)
                continue
            cls_key = value.pop('name')
            name = ConfigType(cls_key)
            config_class = TYPE_TO_CONFIG_CLS[name]
            cls_dict[key] = config_class.from_dict(value)
        trainer_name = TrainerConfigType(trainerType)
        trainer_cls = TYPE_TO_TRAINER_CLS[trainer_name]
        return trainer_cls.from_dict( cls_dict )
            
    # def get_num_warmup_steps(self, warmup_ratio):
    #     len_dataloader = len(self.train_dataloader)
    #     num_epochs = self.args.trainer.num_train_epochs

    #     num_update_steps_per_epoch  = 

    def get_scheduler_config(self, scheduler_config_params: dict):
        # 检查是否包含 'name' 属性
        if 'name' not in scheduler_config_params:
             raise ValueError("Scheduler 'name' is required in scheduler_params.")
        
        # 取出 'name' 并从字典中删除
        scheduler_name = scheduler_config_params.pop('name')
        warmup_ratio = scheduler_config_params.pop('warmup_ratio')
        # num_warmup_steps, num_training_steps = self.get_num_warmup_steps(warmup_ratio)#update later
        base_config = {"warmup_ratio":warmup_ratio}
        scheduler_params = scheduler_config_params[scheduler_name]
        scheduler_params.update(base_config)

        name = ScheType(scheduler_name)
        config_class = TYPE_TO_SCHEDULER_CFG_CLS[name]
        if not config_class:
             raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")
        
        # 动态实例化该调度器的配置类
        return config_class.from_dict(scheduler_params)


