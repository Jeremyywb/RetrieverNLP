来源：https://zhuanlan.zhihu.com/p/809229182
作者：ybq

数据多样性
经历了一年多的磕磕绊绊，目前的 LLM 从业人员大多都会认同：sft 训练数据的核心是数据多样性和数据质量，数据数量并不重要。

数据质量就不谈了，prompt 可以不那么严谨，能看懂就行，但 answer 是尽量一个标点符号都不要有错误的，该中文引号就中文引号，该单引号就单引号，该把 GPT4 啰哩啰嗦的回复精简一下就精简。

我们重点说说数据多样性。即使到了今天，也没人能定义清楚说怎样的一份训练数据叫做数据多样性足够好。我们能做的只能是从先验的角度，把模型能遇到的各种任务类型都让它见一次。从个人经验来说，我认为数据多样性主要包含两个维度，“数据用途”和“数据形式”。

先说数据用途，也就是 task_type，可以结合这几个思路进行数据收集：

- 1. OpenAI 官网列出了 ChatGPT 擅长的所有任务项，诸如翻译、emoji 聊天……之类的。我们就每个任务项都想办法来一点数据，照着尖子生的作业抄；
- 2. LLM 毕竟是个语言模型，传统的每个 NLP 模型它都应该能胜任，那就把什么 NER、机器阅读理解、意图识别等传统的 NLP 任务也给模型补充一点，如果已有类似任务就不补充了。训练数据也很好搞，传统 NLP 数据集质量都很高，直接拿来用就行；
- 3. 参考业务需求，下游业务需要某个特殊场景的任务，那就让 sft 阶段提前见一见，这种数据的典型代表就是过年前给模型灌一些对春联、猜灯谜的的数据。只要数据质量没问题，一般都不会破坏模型能力；
- 4. ……
重点来了，每一条 sft 训练数据必须要 task_type 类型，千万别搞大杂烩，否则对后续的 case 分析简直是灾难性的伤害。在实际工作中，双层 task_type 都很常见，比如“逻辑推理 - 常识推理”，“逻辑推理 - cot 多步骤推理” 这种。至于每种 task_type 的数据量，别搞平均主义：难 task_type 酒数据多点，简单 task_type 就数据少点，也要结合自己的 base 模型能力动态调整。

task_type 的划分就是 sft 数据最重要的基建工作，没有之一。

我们还需要从数据形式的角度来兼顾数据的多样性：


- prompt 表达方式多样性，不要千篇一律的“把中文句子 A 翻译成英文”，也要适当有一些“我在英国旅游，我现在需要向路人问路，我想表达 A 的意思，该怎么说”，“我是一个英文老师，我需要向我的学生讲解句子 A 用英文怎么写，请你用最正宗的表达方式帮我完成。”这么做的目的是防止模型只认识 prompt 中的几个关键 token，进而导致训练过拟合或者泛化性变差；
- prompt 长度均衡，既要有短数据，也要有长数据，避免模型的 attention 退化到无法聚焦长 prompt。长数据还不能是字面意思的长，要有那种关键信息藏在 开头 / 中间 / 结尾 的各种数据场景，避免模型在训练时偷懒，只对 prompt 的起始 token 或结束 token 有 attention；
- answer 长度均衡，不能让模型没出输几个 token 就停止，适当的有一些语料让它学会输出尽量长的 answer，否则模型会很难 follow “不少于2000字” 这种指令；
- 多轮聊天的切换 topic 能力，也就是说，有的数据当前 query 是和 session 有关系的，有的数据则是当前 query 和 session 毫无关系，要让模型自己学会判断 query 是否和 session 有关。类似的数据还要有 system 是否生效，有些数据 system 是个摆设，有些数据的 answer 则和 system 直接相关；
- answer 分布的多样性，这最重要，千万别总共一万条训练数据，一千条数据的 answer 都说同一句话，answer 可是算 loss 的，太单一的话会严重让模型过拟合；
- ……
概括起来，所有的数据形式多样性都可以总结为一句话： 数据形式不能让模型找到规律，关键信息在 prompt 中的位置分布要足够随机。目的是避免模型在训练时退化，只聚焦于某些或某些位置的 token，而不是聚焦于完整的 prompt。模型和人一样，骨子里都是有偷懒倾向的。 


数据生产
生产 prompt

说实话，我已经不太记得通用模型的 prompt 是怎么造的了，那都是去年的工作，感觉当时都是直接翻译英文数据集的 prompt 并重新标注完成的。印象里，斯坦福有一个 self-Instruct 的工作，给每个 task_type 准备一些 seed prompt，然后随机采样 seed，在喂给一个能力很强的 pretrain 模型，让它基于这些 seed 问题再续写出一些问题。其实也不必是 pretrain 模型，GPT4 模型的指令 follow 能力已经足够强了，让它基于一些 seed 问题直接仿写出一些 prompt 也是可以的。

今年的话，应该有很多现成的 sft 训练集，或者是 nlp 训练集，想个办法到处搜刮一下，然后简单筛选下质量就行，反正我们只要 prompt，并不要 answer。最近讨论的比较热的“合成数据”，基本也都是各种启发式规则造 prompt，可以重点留意一下。按照我前文中介绍的数据多样性，去搜集不同 task_type 的数据集集合，然后适当做做改写。实在是找不到合适的 prompt ，就自己动手写一点，answer 写不出来，prompt 还能写不出来吗？

特别要注意，收集或设计 prompt 的时候一定要结合实际情况，不要指望模型一次性写一篇万字爽文，这种事情连人都做不到。我们要把比较困难的任务提前拆解好 prompt ，比如：

- prompt1 ：请设计一个重生故事的大纲，大纲包含“父母重男轻女，女主高考状元，弟弟彩礼”等要素；
- prompt2 ：请基于给定的故事大纲，扩充内容，生成一篇不少于多少字的文章。
LLM 只是知识量比人多，而不是知识掌握度比人精细。如果普通人做起来都费劲，那这个 prompt 大概率是需要拆解的，这在“利用 sft 后的模型去对接业务”时格外重要。


训练目录示例

```
code/
├── reasoner/                 # COT训练代码目录
│   ├── sft_dataset.py        # 自定义 dataset
│   ├── sft_loader.py         # 自定义 collator
│   ├── sft_optimizer.py      # 自定义 optimizer
│   └── sft_trainer.py        # 自定义 trainer
│
├── conf/            # 配置文件目录
│   └── qwen_reasoner/        # reasoner 配置目录
│       └──conf_reasoner_7b.yaml #训练配置目录
│
└── train_eedi_sft.py         # SFT 训练主入口
```