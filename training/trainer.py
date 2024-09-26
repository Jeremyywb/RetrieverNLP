# This is the trainer.py file in training folder

        # model: Union[PreTrainedModel, nn.Module] = None,
        # args: TrainingArguments = None,
        # data_collator: Optional[DataCollator] = None,
        # train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        # eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        # tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # model_init: Optional[Callable[[], PreTrainedModel]] = None,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # callbacks: Optional[List[TrainerCallback]] = None,
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] =

# preprocess_logits_for_metrics
# gather_function
# evaluation_loop
# 既然是trainer 不存在 没有label情况...除非无监督？当前不考虑

# metrics = self.compute_metrics(
#                             EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
#                             compute_result=is_last_step,
#                         )


# TrainOutput(self.state.global_step, train_loss, metrics)
# _inner_training_loop
# 
# loss.detach() / self.args.gradient_accumulation_steps
# ==>training_step
# from FlagEmbedding import FlagModel


#  bge的模型   model = BiEncoderModel(model_name=model_args.model_name_or_path,
#                            normlized=training_args.normlized,
#                            sentence_pooling_method=training_args.sentence_pooling_method,
#                            negatives_cross_device=training_args.negatives_cross_device,
#                            temperature=training_args.temperature,
#                            use_inbatch_neg=training_args.use_inbatch_neg,
#                            )