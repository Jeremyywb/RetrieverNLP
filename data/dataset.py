
import csv
import json
import torch
import random
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..config.configs import BaseConfig
from transformers     import PreTrainedTokenizer
from tqdm.auto        import tqdm
from abc              import ABC, abstractmethod
# from torch.utils.data import DataLoader,RandomSampler


class BaseNLPDataset(Dataset, ABC):
    def __init__(self, data, config: BaseConfig, tokenizer:PreTrainedTokenizer):
        """
        初始化数据集，并根据配置文件处理数据
        
        参数:
        - data: 数据集的原始数据 (可以是 JSON 或 CSV 格式数据)
        - config: 包含所有参数信息的 BaseConfig 类实例
        """
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.samples = []  # 存放处理后的样本
        self.create_samples()

    @abstractmethod
    def create_samples(self):
        """
        处理样本的方法，子类可以覆盖此方法进行自定义数据处理逻辑
        self.samples.append(each_sample)
        """
        # 示例处理方法
        pass

    @classmethod
    def load_from_json(cls, config: BaseConfig, tokenizer:PreTrainedTokenizer):
        """
        从 JSON 文件加载数据集，文件格式为一行一个 dict

        参数:
        - json_path: JSON 文件路径
        - config: 包含配置信息的 BaseConfig 实例
        
        返回:
        - 类实例
        """
        data = []
        with open(config.json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))  # 每行一个 JSON 对象
        return cls(data, config, tokenizer)

    @classmethod
    def load_from_csv(cls, config: BaseConfig, tokenizer:PreTrainedTokenizer):
        """
        从 CSV 文件加载数据集

        参数:
        - csv_path: CSV 文件路径
        - config: 包含配置信息的 BaseConfig 实例

        返回:
        - 类实例
        """
        data = []
        with open(config.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)  # 将每一行转为字典并存入列表
        
        return cls(data, config, tokenizer)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        """
        return self.samples[idx]

    def __len__(self):
        """
        返回样本总数
        """
        return len(self.samples)



class BgeRetrieverDataset(BaseNLPDataset):
    def create_samples(self):
        """
        BgeRetrieverData 的自定义样本处理逻辑
        """
        for item in self.data:
            query = item['query']
            if not isinstance(query, list):
                query = [query]
            if self.config.query_instruction_for_retrieval is not None:
                query = [self.args.query_instruction_for_retrieval + q for q in query]

            
            pos_passage = item['pos']
            neg_passage = item['neg']
            assert isinstance(pos_passage, list)
            pos = random.choice(pos_passage)

            for i in range(0, len(neg_passage), self.config.train_group_size-1):
                neg_group = neg_passage[i:i+self.config.train_group_size-1]
                if len(neg_group) < self.config.train_group_size - 1:
                    continue
                passages = []
                passages.append(pos)
                passages.extend(neg_group)

                if self.config.passage_instruction_for_retrieval is not None:
                    passages = [self.config.passage_instruction_for_retrieval+p for p in passages]
                
                _query = self.prepare_tokens(self.tokenizer,  query, self.config.query_max_len )
                _passages = self.prepare_tokens( self.tokenizer,   passages, self.config.passage_max_len )
                self.samples.append({
                    'query':_query,
                    "passages":_passages,
                    "passag_id": torch.tensor([item['passag_id']], dtype = torch.long)
                    })

    def prepare_tokens(self, tokenizer, texts, max_len):
        inputs = tokenizer.batch_encode_plus(
                    texts,
                    padding        = 'max_length',
                    return_tensors = 'pt',
                    max_length     = max_len,
                    truncation     = True
                )
        return {k:torch.tensor(v, dtype = torch.long) for k,v  in inputs.items()}
    



class BgeRetrieverEvalDataset(BaseNLPDataset):
    def create_samples(self):
        """
        BgeRetrieverData 的自定义样本处理逻辑
        file format: {
                "query": "query", 
                "pos":["pos1","pos2","pos3"...], 
                "neg":["neg1","neg2","neg3"...],
                "passag_id": passag_id
            }
        """

        for item in self.data:
            query = item['query']
            if not isinstance(query, list):
                query = [query]
            if self.config.query_instruction_for_retrieval is not None:
                query = [self.args.query_instruction_for_retrieval + q for q in query]

            passages = []
            pos_passage = item['pos']
            neg_passage = item['neg']

            assert isinstance(pos_passage, list)
            pos = random.choice(pos_passage)
            passages.append(pos)

            if len(neg_passage) < self.config.train_group_size - 1:
                num = math.ceil((self.config.train_group_size - 1) / len(item['neg']))
                negs = random.sample(item['neg'] * num, self.config.train_group_size - 1)
            else:
                negs = random.sample(item['neg'], self.config.train_group_size - 1)
            passages.extend(negs)

            if self.config.passage_instruction_for_retrieval is not None:
                passages = [self.config.passage_instruction_for_retrieval+p for p in passages]
            
            _query = self.prepare_tokens(self.tokenizer,  query, self.config.query_max_len )
            _passages = self.prepare_tokens( self.tokenizer,   passages, self.config.passage_max_len )
            self.samples.append({
                'query':_query,
                "passages":_passages,
                "passag_id": torch.tensor([item['passag_id']], dtype = torch.long)}
            )



    def prepare_tokens(self, tokenizer, texts, max_len):
        inputs = tokenizer.batch_encode_plus(
                    texts,
                    padding        = 'max_length',
                    return_tensors = 'pt',
                    max_length     = max_len,
                    truncation     = True
                )
        return {k:torch.tensor(v, dtype = torch.long) for k,v  in inputs.items()}
    

