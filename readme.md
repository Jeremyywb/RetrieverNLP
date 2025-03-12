# RetrieverNLP

## 项目简介

RetrieverNLP 是一个用于检索类模型微调的简单通用框架，旨在帮助开发者和研究人员快速搭建、训练、微调针对信息检索任务的模型。本项目参考transformers库实现trainer模块，并在此基础上进行修改，以适应检索类模型训练需求，旨在便于二次开发。

## 技术栈

- Python
- PyTorch
- Transformers (Hugging Face)


## 安装指南

1. 克隆项目仓库：
   ```bash
   git clone https://github.com/Jeremyywb/RetrieverNLP.git
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## Trainer
  - training_config_yaml_or_dict: 训练配置文件路径或者字典 reference :  resource/args-yaml/bge-emebeding-training-config-V1.yaml
  - model_name_or_instance: 
  - args
  - data_collator
  - train_dataset
  - eval_dataset
  - tokenizer
  - model_init
  - compute_metrics
  - callbacks
  - optimizers
  - preprocess_logits_for_metrics

### model_name_or_instance 
   - instance/实例化模型 
   - name/本框架支持的模型名称 reference :
   - 区别于 model_name_or_path（预训练模型作为backbone） 
TYPE_TO_MODEL_CLS:{
   ModelType.BgeEmbedding:BgeBiEncoderModel,
   ...
}

### set_all_config
配置所有参数
   - TYPE_TO_CONFIG_CLS 模型参数类名称到参数类名映射，当前实现9种训练相关参数配置
   - TYPE_TO_CONFIG_CLS中 SchedulerConfig 为空，根据名字选择需要的Scheduler而选择相应config，参考transformers实现

### Eval strategy
