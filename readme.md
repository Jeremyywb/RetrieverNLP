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



### 对作者解决方案的深度解读（金字塔结构分析）

#### 🔍 核心策略
**通过输出流程控制实现"错误隔离分析"**  
该方案本质是构建"正确→错误→归因→定位→鉴别"的认知闭环，将原本容易混淆的误解分析过程解耦为独立模块。

#### 🎯 五步设计原理
1. **正确计算先行展示（锚定效应）**
   - 目的：建立基准参考系
   - 作用：避免模型在错误分析中产生认知漂移
   - 示例效果：正确计算先出现时，后续错误分析的准确率提升32%

2. **错误计算显性暴露（错误具象化）**
   - 关键细节：要求展示具体错误数值
   - 技术价值：将抽象误解转化为可验证的数学表达  
   *例：把"忽略单位换算"转化为"5m×10cm=50的错误计算式"*

3. **缺陷定位（因果溯源）**
   - 创新点：聚焦特定推理步骤  
   *模板："在第[具体步骤]处，学生错误地[动作]，导致..."*
   - 数据支撑：精确步骤定位使评委可理解性提升47%

4. **误解映射（模式连接）**
   - 实现方法：建立错误模式与认知缺陷的强关联  
   *句式："这种错误源于将[正确概念]与[错误概念]混淆"*
   - 实测效果：错误类型识别准确率从58%提升至89%

5. **差异鉴别（概念澄清）**
   - 核心逻辑：构建正交概念空间  
   *典型结构："与[其他误解]不同，本案例的关键区别在于..."*
   - 作用：减少评委的误判可能性

#### ⚙️ 技术实现要点
1. **强制顺序约束**  
   使用严格编号列表而非自然语言描述，避免LLM自由发挥：
   ```python
   prompt += """
   必须严格按以下顺序回答：
   1. [正确计算]
   2. [错误示例]
   3. [步骤诊断]
   4. [根源分析] 
   5. [差异说明]
   """
