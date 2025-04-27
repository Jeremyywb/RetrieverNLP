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
from ..data.dataset import BgeRetrieverDataset,BgeRerankerDataset,BgeRerankerEvalDataset
from ..models.retriever import BgeBiEncoderModel
from ..models.reranker import BgeCrossEncoder
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
        ,RerankerValDataConfig

)
from collections.abc import Mapping
from torch.cuda import amp
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
        DatasetType.RERANKERTRAIN:BgeRerankerDataset,
        DatasetType.RERANKERVALID:BgeRerankerEvalDataset,

}

TYPE_TO_CONFIG_CLS = {
        ConfigType.TRAINSET:RetrieverDataConfig,
        ConfigType.VALIDSET:RetrieverDataConfig,
        ConfigType.TESTSET:RetrieverDataConfig,
        ConfigType.BGERETRIEVERTRAIN:RetrieverDataConfig,
        ConfigType.BGERETRIEVERVALID:RetrieverDataConfig,
        ConfigType.BGERETRIEVERTEST:RetrieverDataConfig,
        ConfigType.RERANKERTRAIN:RerankerDataConfig,
        ConfigType.RERANKERVALID:RerankerValDataConfig,
        ConfigType.TRAINLOADER:DataLoaderConfig,
        ConfigType.VALIDLOADER:DataLoaderConfig,
        ConfigType.TESTLOADER:DataLoaderConfig,
        ConfigType.BGEEMBEDDING:RetrieverModelConfig,
        ConfigType.BGERERANKER:RerankerModelConfig,
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
        ModelType.BGERERANKER:BgeCrossEncoder

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
        logger.info("***** Init Trainer *****")
        self.compute_metrics = compute_metrics
        logger.info(f"Init Trainer  : Configurations Started")
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        if args is None:
            if isinstance(training_config_yaml_or_dict, str):
                config_dicts = load_yaml(training_config_yaml_or_dict)
                self.args = self.set_all_config(config_dicts)
            else:
                self.args = self.set_all_config( training_config_yaml_or_dict )
        else:
            self.args = args
        logger.info(f"Init Trainer  : Configurations Finished")
        print(self.args)

        self.accelerator = None
        self.is_deepspeed_enabled = False
        self.is_local_process_zero = True
        self.is_world_process_zero = True

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

        logger.info(f"Init Trainer  : Tokenizer Started")
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
        logger.info(f"Init Trainer  : Tokenizer Finished")

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

        logger.info(f"Init Trainer  : Dataset Started")
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
        
        logger.info(f"Init Trainer  : Dataset Finished")


        if data_collator is None:
            raise ValueError("data_collator is required")
        if self.args.trainer.task == 'retrieval':
            self.data_collator = data_collator(self.tokenizer)
        else:
            self.data_collator = data_collator

        if isinstance(model_name_or_instance, str):
            name = ModelType(self.args.model.model_type)
            _cls = TYPE_TO_MODEL_CLS[name]
            self.model = _cls(self.args.model)
        else:
            self.model = model_name_or_instance

        logger.info(f"Init Trainer  : Model Freeze Started")
        if self.args.trainer.num_freeze_layers is not None:
            self.model.freeze_layers(self.args.trainer.num_freeze_layers)
        logger.info(f"Init Trainer  : Model Freeze Finished")
        self.model.to(self.device)

        logger.info(f"Init Trainer  : Dataloader Started")
        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()
        logger.info(f"Init Trainer  : Dataloader Finished")

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


        logger.info(f"Init Trainer  : Optimizer Started")
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.get_optimizer_and_scheduler()
        logger.info(f"Init Trainer  : Optimizer Finished")

        logger.info(f"Init Trainer  : Callbacks Started")
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.args.callbacks.early_stopping_patience,
            early_stopping_threshold=self.args.callbacks.early_stopping_threshold
        )
        default_callbacks = [DefaultFlowCallback,PrinterCallback,ProgressCallback, early_stopping_callback ]
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        logger.info(f"Init Trainer  : Callbacks Finished")

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero,
            is_world_process_zero=self.is_world_process_zero,
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )


        # use_amp  not implemented yet

        self.current_flos = 0
        self.control = self.callback_handler.on_init_end(self.args.callbacks, self.state, self.control)

        self._train_batch_size = self.args.train_dataloader.batch_size
        logger.info(f"Init Trainer  : Trainer Finished waiting 5 seconds...")
        time.sleep(5)
        logger.info(f"Init Trainer  : Trainer Ended")

    def train(self):
        # self.callback_handler.on_train_begin(self.args.callbacks, self.state, self.control)
        # self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self.state.is_hyper_param_search = False
        self.state.train_batch_size = self.args.train_dataloader.batch_size
        self.state.is_epoch_progress_bar_enabled = self.args.callbacks.is_epoch_progress_bar_enabled

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
        logger.info(f"  Num examples       = {num_examples:,}")
        logger.info(f"  Num Epochs         = {self.args.trainer.num_train_epochs:,}")
        logger.info(f"  Train Batch Size   = {self.args.train_dataloader.batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.args.trainer.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps    = {self.num_training_steps:,}")
        self.state.epoch = 0
        self.state.global_step = 0
        self.state.max_steps = self.num_training_steps
        self.state.num_train_epochs = self.args.trainer.num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True
        self.state.no_prediction_bar = True

        start_time = time.time()

        tr_loss = torch.tensor(0.0).to(self.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.model.zero_grad()
        self.optimizer.zero_grad()

        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(self.args.callbacks, self.state, self.control)
        total_batched_samples = 0
        if self.args.trainer.fp16:
            scaler = amp.GradScaler(enabled=True)
        for epoch in range(self.args.trainer.num_train_epochs):
            steps_in_epoch = len(self.train_dataloader)
            self.control = self.callback_handler.on_epoch_begin(self.args.callbacks, self.state, self.control)
            # self.train_dataloader.set_epoch(epoch)
            for step, batch in enumerate(self.train_dataloader):
                total_batched_samples += 1
                
                tr_loss_step = self.training_step(batch, scaler)
                if tr_loss.device != tr_loss_step.device:
                    raise ValueError("tr_loss and tr_loss_step should be on the same device")
                tr_loss += tr_loss_step
                # if (self.state.global_step+1) % self.args.callbacks.logging_steps == 0:
                #     self.log_metrics()

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch<= self.accumulation_steps and (step+1) == steps_in_epoch
                )
                if (
                    total_batched_samples % self.accumulation_steps == 0 or
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    if self.args.trainer.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.trainer.max_grad_norm)
                    
                    self.control = self.callback_handler.on_pre_optimizer_step(self.args.callbacks, self.state, self.control)
                    logger.info( " At Optimizer Stepping")

                    if self.args.trainer.fp16:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(self.args.callbacks, self.state, self.control)
                    
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step+1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args.callbacks, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm)
                else:
                    self.control = self.callback_handler.on_substep_end(self.args.callbacks, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args.callbacks, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm)

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
            kwargs = {"device": self.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
    

    def training_step(self,batch,scaler):
        self.model.train()
        if self.args.trainer.task == 'retrieval':
            batch.pop('passag_id')
        self.batch_infos = self.__describe(batch)
        inputs = self._prepare_input(batch)
        if self.args.trainer.fp16:
            with amp.autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss = loss / self.args.trainer.gradient_accumulation_steps
        else:
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss = loss / self.args.trainer.gradient_accumulation_steps
        if self.args.trainer.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        del inputs, outputs
        if self.args.trainer.empty_cache_original_step:
            torch.cuda.empty_cache()

        if self.args.trainer.torch_empty_cache_steps is not None:
            if (self.state.global_step+1) % self.args.trainer.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()
        return loss

    def __describe(self, batch):
        def analyze_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.shape
            elif isinstance(tensor, (list, tuple)):
                return tuple(analyze_tensor(t) for t in tensor)
            elif isinstance(tensor, dict):
                return {k: analyze_tensor(v) for k, v in tensor.items()}
            else:
                return None

        def recursive_analyze(data, prefix=''):
            batch_info = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    new_prefix = f"{prefix}_{k}" if prefix else k
                    batch_info.update(recursive_analyze(v, new_prefix))
            else:
                shape = analyze_tensor(data)
                if shape is not None:
                    batch_info[prefix] = shape
            return batch_info

        return recursive_analyze(batch)

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
            self.control = self.callback_handler.on_evaluate(self.args.callbacks, self.state, self.control, metrics)
        
        if self.control.should_save:
            self._save_checkpoint(metrics)
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
        
        self.log(output.metrics)
        return output.metrics
    
    def predict_passages(self,passages):
        self.model.eval()
        passages_inputs = self.train_dataset.prepare_tokens(self.tokenizer, passages, self.args.trainset.passage_max_len)
        all_psg_preds = []
        psg_batch_size = max(self.args.train_dataloader.batch_size//2,1)
        for i in range(0,len(passages),psg_batch_size):
            batch = {k:v[i:i+psg_batch_size] for k,v in passages_inputs.items()}
            _max_length = batch['attention_mask'].sum(dim=1).max().item()
            for k,v in batch.items():
                batch[k] = v[:,:_max_length]
            batch = self._prepare_input(batch)
            with torch.no_grad():
                outputs = self.model.encode_passages(batch)
            all_psg_preds.append(outputs.cpu())
            del batch, outputs
            torch.cuda.empty_cache()
        ouptut = torch.cat(all_psg_preds,dim=0)
        del all_psg_preds
        return ouptut


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
                observed_batch_size = logits.shape[0]
                observed_num_samples += observed_batch_size
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
        if self.args.trainer.task == 'reranker':
            labels = inputs.pop('pos_mask')
            # labels (batch_size, num_doc)-> target_for_loss (batch_size,1 ) pos index
            target_for_loss = torch.argmax(labels, dim=1)
            
        inputs = self._prepare_input(inputs)
        with torch.no_grad():
            if self.args.trainer.task == 'retrieval':
                outputs = self.model(**inputs)
            elif self.args.trainer.task == 'reranker':
                outputs = self.model.encode( 
                    self.args.valid_dataloader.batch_size, 
                    self.args.validset.group_size,
                    inputs['inputs'], target_for_loss.to(self.device) 
                )
        if self.args.trainer.task == 'retrieval':
            logits = outputs.q_reps
        else:
            logits = outputs.logits
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
        

        if metrics is not None and self.args.callbacks.metric_for_best_model is not None:
            metric_to_check = self.args.callbacks.metric_for_best_model
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
                print(f"model improved ! from {self.state.best_metric} to {metric_value} ")
                self.state.best_metric = metric_value

                if os.path.exists(self.best_model_previous):
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
                self.model.save(output_dir)
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
        self.control = self.callback_handler.on_log(self.args.callbacks, self.state, self.control, logs)


    def get_optimizer_and_scheduler(self):
        # print(self.args.optim)
        # if self.args.trainer.num_freeze_layers is not None:
        #     layers = list(self.model.backbone.encoder.layer[ self.args.trainer.num_freeze_layers:])
        #     print(f"total layers: {len(list(self.model.backbone.encoder.layer))}, freeze layers: {self.args.trainer.num_freeze_layers}")
        # else:
        #     layers = [self.model.backbone.embeddings] + list(self.model.backbone.encoder.layer)
        # layers.reverse()


            # 动态获取 encoder 和 embeddings，适配不同的模型
        if hasattr(self.model.backbone, 'roberta'):  # 适配 XLMRobertaForSequenceClassification
            encoder = self.model.backbone.roberta.encoder.layer
            embeddings = self.model.backbone.roberta.embeddings
            head = self.model.backbone.classifier  # 分类头
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
            encoder = self.model.backbone.encoder.layer
            embeddings = self.model.backbone.embeddings
            head = getattr(self.model.backbone, 'classifier', None)  # 检查是否有 classifier
        else:
            raise ValueError("Model structure not supported. Please check your model's encoder, embedding, and head structure.")

        # 判断是否有冻结层的要求
        if self.args.trainer.num_freeze_layers is not None:
            layers = list(encoder[self.args.trainer.num_freeze_layers:])
            print(f"total layers: {len(list(encoder))}, freeze layers: {self.args.trainer.num_freeze_layers}")
        else:
            layers = [embeddings] + list(encoder)

        # 如果有分类头（head），则将其加入最后一个层级
        if head:
            layers.append(head)
        
        # 倒序排列层，分层学习率
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
        lr_scheduler_type = lr_scheduler_kwargs.pop('name')
        lr_scheduler_kwargs.pop('warmup_ratio')
        # lr_scheduler_kwargs.update({'num_warmup_steps':self.num_warmup_steps,'num_training_steps':self.num_training_steps})
        if self.lr_scheduler is None:
            lr_scheduler = get_scheduler(
                lr_scheduler_type,
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
        scheduler_name = scheduler_config_params['name']
        warmup_ratio = scheduler_config_params.pop('warmup_ratio')
        # num_warmup_steps, num_training_steps = self.get_num_warmup_steps(warmup_ratio)#update later
        base_config = {"warmup_ratio":warmup_ratio}
        scheduler_params = scheduler_config_params[scheduler_name]
        scheduler_params.update(base_config)
        scheduler_params['name'] = scheduler_name
        name = ScheType(scheduler_name)
        config_class = TYPE_TO_SCHEDULER_CFG_CLS[name]
        if not config_class:
             raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")
        
        # 动态实例化该调度器的配置类
        return config_class.from_dict(scheduler_params)



你是一个出色的算法工程师。给你数据描述，以及任务指导，同时还有一份实践代码。完成一份更加高效的代码

以下是相关的数据描述：
数据1 dataframe（多选题数据，原多选题拆解而来，拆解逻辑为多选题题目id+错误选项的名称A也即,B,C,D）：
查询数据｛
query_id:查询id，唯一值，其构成为 original_query_id （是字符串，多选题的题目的id）+ "下划线" + 答案选项（A,B,C,D中的一个）,
content_id: 正样本id， 
MisconceptionName:正样本文本,
QuestionText: 查询文本,
CorrectAnswerText: 正确答案文本,
InCorrectAnswerText：错误答案文本,
Explanation: cot数据
｝
数据2 dataframe：
样本池{
    content_id:唯一值 其他所有content id都可以在这个字段找到，
    MisconceptionName:正样本标签文本,
}
数据3 json：#数据来源 查询数据dataframe1的每个query_id对应内容构造prompt通过API获得COT
    额外cot数据{
        "query_id":查询id,
        "content_id":样本id,
        "Explanation":其他API获取的额外cot数据
    }
数据4 json:
    semi 中等难度负样本}{
        "query_id":查询 id，
        "content_ids":10个中等难度样本的content_id 逗号分隔的拼接文本 如 22,33,22,33...   
    }
数据5 json:
    hard 困难难度负样本}{
        "query_id":查询 id，
        "content_ids":10个困难难度样本的content_id 逗号分隔的拼接文本 如 22,33,22,33...   
    }



设计dataset，以简单为主。
1. 一个统一可用的dataset，旨在可以对于查询文本、样本标签 MisconceptionName 文本 也即content的tokenized，以及cot文本进行tokenized，但是需要注意保留 query_id与每个dataset的item对应关系
2. 减少计算量，所有样本content_id的tokenize通过样本池的tokenize关联获取
3. 用于train的dataset，以query_id作为主键（但是注意他是文本，可能不能直接用到dataloader），收集其对应的 
- 当前query_id下，QuestionText 的  tokenized
- 当前query_id下，positive 正样本对应 的 MisconceptionName 标签 的tokenized
- 当前query_id下，两个 cot （ 查询数据本身带有的cot以及额外的cot） 对应的 tokenized
- 当模式为 easy 时，要求到此为止
- 当模式为 semi 时，要求添加 当前 query_id 对应的 中等难度负样本，其tokenized
- 当模式为 hard 时，要求添加 当前 query_id 对应的 中等难度负样本，其tokenized


以下是相关的任务指导：
## 四个基础dataset实例
- query_dataset： query_id为属性，因为输入是 dataframe(query_id, QuestionText),每个item（QuestionText的tokenized ）的id索引和query_id索引一致。 建议：额外生成一个属性字典，映射 query_id到 content_id 的对应关系
- content_dataset: content_id(从0升序的自然数) 作为属性，因为输入是 dataframe(content_id, MisconceptionName),每个item（QMisconceptionName的tokenized )的id索引和content_id一致
- cot_dataset: query_id 作为属性，因为输入是 dataframe(query_id, Explanation),每个item（ Explanation 的tokenized )的id索引和 query_id 一致。 建议：额外生成一个属性字典，映射 query_id到dataset索引的对应关系
- exteral_cot_dataset 额外的cot: query_id 作为属性，因为输入是 dataframe(query_id, Explanation),每个item（ Explanation 的tokenized )的id索引和 query_id 一致。 建议：额外生成一个属性字典，映射 query_id到dataset索引的对应关系
这四个dataset可以使用同一个 dataset类实例化
## 用于train的dataset
### 当前模式为 easy 情况下
输入为4个dataset，以 query_dataset为主，其query_id作为查询其他数据的键，
对于 **getitem**函数，
- 1. query_id = self.query_id[idx]
- 2. 以 query_id 为查询键， 获取 query_dataset中 content_id，然后其 tokenized 从 content_dataset获取
- 3. 以 query_id 为查询键，查询 cot_dataset 中， 对应query_id的 item_id， 获取cot 的 tokenized
- 4. 以 query_id 为查询键，查询 exteral_cot_dataset 中， 对应query_id的 item_id， 获取额外 cot 的 tokenized
### 当前模式为 semi 情况下
- 注意生成 query_id_to_semi_content_id 映射关系
对于 **getitem**函数，
- 前4个步骤与 easy 模式相同
- 5. 以 query_id 为查询键，从 query_id_to_semi_content_id 获取当前所有 content_id ，通过文本分割生成 content_id 列表
- 6. 基于 content_id 获取 列表中所有 content_id 在 content_dataset 中的 tokenized
### 当前模式为 hard 情况下
与 semi 模式相同，但是用的是hard数据




你是一个优秀的模型微调工程师。现在，以下内容与之前的数据一致，需要完成以下的内容要求
## hard 样本生成
生成 hard 样本 需要生成10个，生成步骤为 ：
- 最难负样本： 同一道多选题 其他选项的（不含正确选项，数据中也没有） content_id，也即相同 oringal_query_id的不同 content_id
- 基于content相似： 与当前样本 content 相似度高于阈值的 content 选择不超过 3个
- 基于query相似: 与当前样本 query文本（不同题目） 相似度高于阈值的 content 选择不超过 2个
- 当前query去content池检索， 筛选剔除掉之前已经得到的content_id以及与当前content_id一样的，基于总目标10个，当前需要获取的个数从 topk中获取

### 最难负样本
- 使用【数据1 dataframe】也即query数据，使用字段 query_id 和 content_id
- 获取 原始 oringal_query_id： 现有 query_id实际为结构为 【oringal_query_id +"下划线"+ 选项也即A,D,C,D】，分解得到 oringal_query_id
- 当前样本 相同的 oringal_query_id，其他的content_id作为最难负样本，需要去重且剔除与本 id相同的


### 基于content相似
- 使用【数据1 dataframe】也即query数据 和【数据2 dataframe】也即样本池
- 使用当前样本的， content作为query，去content池 检索样本
- 样本中排除当前content以及最难负样本，选择 top3 的样本

### 基于query相似

使用当前题目作为query也即 QuestionText ，检索其他题目的 QuestionText,获取top 相似的 QuestionText ，并从所在题目中随机选择2个content id，注意需要剔除已经有的id或者与当前 样本的content_id一样的id，下面是推荐步骤：
- 使用【数据1 dataframe】也即query数据，记录当前样本的一些内容，也即 original_query_id,query_id,content_id,QuestionText
- 生成以 original_query_id 为键的字典记为 oriqid_based_content_dict ，其内容为｛ original_query_id:{ QuestionText:XXXXX,content_ids:[content_id1,content_id2,...]  } }
- 以 query_id 为基础，循环所有样本，检索 其 QuestionText ，对其他QuestionText的相似度进行排名 以 original_query_id 排序为主
- 循环获得 original_query_id ，检查其 oriqid_based_content_dict 中 content_id是否符合要求（content_id不是前面步骤获取的，且不与当前content_id一样），直到获取2个content 


### 基于query检索相似

这是最原始的方法，基于query去检索content池 的内容，获取top content
- 使用【数据1 dataframe】也即query数据 和【数据2 dataframe】也即样本池
- query查询字段参考后面代码生成 alltext字段
- 使用alltext字段取检索content池，获得排序，依次分析其content对应的id，是否之前符合条件（content_id不是前面步骤获取的，且不与当前content_id一样），直到获取足够conten_id(总计10个，扣除前面获取的content_id


```python
def _formatting_func(query):
    task_description = """Retrieve the key misconception behind the wrong answer when given a math problem and its incorrect and correct solutions."""

    return f"Instruct: {task_description}\nQuery: {query}"

def _get_query( row):
    query = ""
    query = f"{row['SubjectName']} - {row['ConstructName']}\n"
    query += f"# Question: {row['QuestionText']}\n"
    query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
    query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
    query = _formatting_func(query)
    return query

dataframe['alltext'] = dataframe.apply(_get_query, axis=0)


由于Q到R（查询到答案），存在推理的间隔鸿沟，本项目方案实现通过课程学习来实现，也即先学习简单后学习难的目标。

课程学习是通过三个损失函数的按权重相加实现。因Q到COT以及COT到R相对容易，故模型先学，然后慢慢增加三元组损失的比重

另外，因为负样本的难度我考虑也通过课程学习逐步提升模型能力，负样本难度慢慢增加，模型先学简单负样本后学习复杂困难负样本。

这就增加整体方案的实现，因为两个都要实现课程学习，请帮我构思一条合理的学习思路，来实现方案目标


cross_scale_loss如何理解，另外，我已经打算分开 easy，semi，hard样本的训练，你这个做法？


对于RetrieverDataset我期望你做以下改造
- 只考虑 semi 和 hard 模式，因为 easy模型实践复杂度，所以我剔除了
- result内容只存放item，例如 pos_item ，存放为 "content":pos
- pos_item 和 neg_item 合并，统一存放到 content中，pos_item放第一个，合并方式是内部的 tensor合并，例如 input_id和attention_mask
- 对于 cot 统一合并为一个，合并方式同上一条的方案

对于loss部分，期望做一些调整
- 因为增加了 ext_cot_item相关的数据，我期望计算loss的时候考虑进去

其中loss部分现在的实践方法如下所示:
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNegTripletLoss(nn.Module):
    """
    Triplet loss for one positive and multiple negatives per anchor.
    Positive is contents[:,0,:], negatives are contents[:,1:,:].
    L = mean(max(d(a,p) - d(a,n_i) + margin, 0)) averaged over all negatives and batch.
    """
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor: torch.Tensor, contents: torch.Tensor) -> torch.Tensor:
        # anchor: [batch_size, hidden_size]
        # contents: [batch_size, num_contents, hidden_size]
        pos = contents[:, 0, :]              # [batch_size, hidden_size]
        negs = contents[:, 1:, :]            # [batch_size, num_negatives, hidden_size]

        # compute distances
        d_pos = torch.norm(anchor - pos, p=self.p, dim=1)                   # [batch_size]
        # expand anchor for negs
        a_exp = anchor.unsqueeze(1).expand_as(negs)                         # [batch_size, num_neg, hidden]
        d_negs = torch.norm(a_exp - negs, p=self.p, dim=2)                  # [batch_size, num_neg]

        # triplet losses per negative
        losses = F.relu(d_pos.unsqueeze(1) - d_negs + self.margin)          # [batch_size, num_neg]
        return losses.mean()   
    



class UnifiedCoTLoss(nn.Module):
    """
    Unified CoT Alignment & Triplet Loss supporting both multi-negative and in-batch negatives.
    Usage:
      - Multi-negative scenario: call forward(anchor, contents=..., cot=...)
      - In-batch scenario: call forward(anchor, positive=..., cot=...)
    CoT loss: alpha*triplet + beta*||anchor-cot||^2 + gamma*||cot-positive||^2
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, margin=1.0, p=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        assert self.alpha+self.beta+self.gamma == 1.0, "alpha + beta + gamma must equal 1.0"
        self.p = p
        self.multi_neg = MultiNegTripletLoss(margin, p)


    def forward(self, anchor: torch.Tensor, cot: torch.Tensor, contents: torch.Tensor=None, positive: torch.Tensor=None) -> torch.Tensor:
        # Compute triplet loss
        l_triplet = self.multi_neg(anchor, contents) if self.alpha > 0 else 0
        # CoT alignment and consistency
        l_align = torch.norm(anchor - cot, p=self.p, dim=1).pow(2).mean()
        l_consis = torch.norm(cot - contents[:, 0, :], p=self.p, dim=1).pow(2).mean()
        return self.alpha * l_triplet + self.beta * l_align + self.gamma * l_consis

```


