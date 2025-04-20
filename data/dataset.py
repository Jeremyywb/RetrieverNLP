
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
    
    def save_json_file(self, data,output_path):
        with open(output_path, 'w') as file:
            for line in data:
                file.write(json.dumps(line) + '\n') # 写入文件 

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
        self.doc_cache = {}

        for i in range(len(self.data)):
            item = self.data[i]

            query = item['query']
            if not isinstance(query, list):
                query = [query]
            if self.config.query_instruction_for_retrieval is not None:
                query = [self.config.query_instruction_for_retrieval +  q for q in query]

            docs        = item['docs']
            pos_mask    = item['pos_mask']
            if 1 in pos_mask:
                pos_index = pos_mask.index(1)
            else:
                pos_index = None

            if pos_index is not None:
                pos = docs.pop(pos_index)
            else:
                pos = item['pos'][0]

            if self.config.contain_inner_neg:
                inner_neg_end = item['inner_neg_end']
            else:
                inner_neg_end = 0
            inner_neg = docs[:inner_neg_end]

            if len(docs) > self.config.group_size - 1:
                docs = inner_neg + docs[self.config.sample_start+inner_neg_end:self.config.sample_end+inner_neg_end]


            #========================================================
            #  original negs
            #====================

            if len(docs) < self.config.group_size - 1:
                num = math.ceil((self.config.group_size - 1) / len( docs ))
                negs = random.sample(docs * num, self.config.group_size - 1)
            else:
                negs = random.sample(docs, self.config.group_size - 1)     
            passages = []
            passages.append(pos)
            passages.extend(negs)

            item['docs'] = negs
            item['pos_mask'] = [0]*(self.config.group_size - 1)
            self.data[i] = item

            if self.config.passage_instruction_for_retrieval is not None:
                passages = [self.config.passage_instruction_for_retrieval+p for p in passages]
            
            _query = self.prepare_tokens(self.tokenizer,  query, self.config.query_max_len )
            _passages = self.prepare_tokens( self.tokenizer,   passages, self.config.passage_max_len )
            self.samples.append({
                'query':_query,
                "passages":_passages,
                "passag_id": torch.tensor([item['passag_id']], dtype = torch.long)}
            )

            #========================================================



            # for i in range(0, len(docs), self.config.group_size-1):
            #     neg_group = docs[i:i+self.config.group_size-1]
            #     if len(neg_group) < self.config.group_size - 1:
            #         continue
            #     passages = []
            #     passages.append(pos)
            #     passages.extend(neg_group)

            #     if self.config.passage_instruction_for_retrieval is not None:
            #         passages = [self.config.passage_instruction_for_retrieval+p for p in passages]
            #     _query = self.prepare_tokens(self.tokenizer,  query, self.config.query_max_len )
            #     _passages = self.prepare_tokens( self.tokenizer,   passages, self.config.passage_max_len )
            #     self.samples.append({
            #         'query':_query,
            #         "passages":_passages,
            #         "passag_id": torch.tensor([item['passag_id']], dtype = torch.long)
            #         })
        
        save_file = self.config.json_path.replace(".json",'_used.json')
        self.save_json_file( self.data, save_file )

    def prepare_tokens(self, tokenizer, texts, max_len):
        _inputs = None
        for text in texts:
            _hash = hash(text)
            if _hash in self.doc_cache:
                _input = self.doc_cache[_hash]
            else:
                _input = tokenizer.encode_plus(
                            text,
                            padding        = 'max_length',
                            return_tensors = 'pt',
                            max_length     = max_len,
                            truncation     = True
                        )
                self.doc_cache[_hash] = _input
            if _inputs is None:
                _inputs = _input
            else:
                _inputs = {k: torch.cat([ _inputs[k], _input[k] ], dim=0) for k in _inputs}
        return {k:torch.tensor(v, dtype = torch.long) for k,v  in _inputs.items()}



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
        self.doc_cache = {}
        for i in range(len(self.data)):
            item = self.data[i]
            query = item['query']
            if not isinstance(query, list):
                query = [query]
            if self.config.query_instruction_for_retrieval is not None:
                query = [self.config.query_instruction_for_retrieval +  q for q in query]

            docs        = item['docs']
            pos_mask    = item['pos_mask']
            if 1 in pos_mask:
                pos_index = pos_mask.index(1)
            else:
                pos_index = None

            if pos_index is not None:
                pos = docs.pop(pos_index)
            else:
                pos = item['pos'][0]

            if self.config.contain_inner_neg:
                inner_neg_end = item['inner_neg_end']
            else:
                inner_neg_end = 0
            inner_neg = docs[:inner_neg_end]
            docs = inner_neg + docs[self.config.sample_start+inner_neg_end:self.config.sample_end+inner_neg_end]

            if len(docs) < self.config.group_size - 1:
                num = math.ceil((self.config.group_size - 1) / len(item['neg']))
                negs = random.sample(docs * num, self.config.group_size - 1)
            else:
                negs = random.sample(docs, self.config.group_size - 1)     
            passages = []
            passages.append(pos)
            passages.extend(negs)

            item['docs'] = negs
            item['pos_mask'] = [0]*(self.config.group_size - 1)
            self.data[i] = item

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
        _inputs = None
        for text in texts:
            _hash = hash(text)
            if _hash in self.doc_cache:
                _input = self.doc_cache[_hash]
            else:
                _input = tokenizer.encode_plus(
                            text,
                            padding        = 'max_length',
                            return_tensors = 'pt',
                            max_length     = max_len,
                            truncation     = True
                        )
            if _inputs is None:
                _inputs = _input
            else:
                _inputs = {k: torch.cat([ _inputs[k], _input[k] ], dim=0) for k in _inputs}
        return {k:torch.tensor(v, dtype = torch.long) for k,v  in _inputs.items()}

class BgeRerankerDataset(BaseNLPDataset):
    def create_samples(self):

        self.query_cache = {}
        self.doc_cache   = {}

        for item in self.data:
            query = item['query']
            if isinstance(query, list):
                query = query[0]
            if self.config.query_instruction_for_retrieval is not None:
                query = self.config.query_instruction_for_retrieval +  query

            docs        = item['docs']
            pos_mask    = item['pos_mask']
            if 1 in pos_mask:
                pos_index = pos_mask.index(1)
            else:
                pos_index = None

            if pos_index is not None:
                pos = docs.pop(pos_index)
            else:
                pos = item['pos'][0]
            if self.config.contain_inner_neg:
                inner_neg_end = item['inner_neg_end']
            else:
                inner_neg_end = 0
            inner_neg = docs[:inner_neg_end]
            docs = inner_neg + docs[self.config.sample_start+inner_neg_end:self.config.sample_end+inner_neg_end]

            if len(docs) < self.config.group_size - 1:
                num = math.ceil((self.config.group_size - 1) / len(docs))
                negs = random.sample(docs * num, self.config.group_size - 1)
            else:
                negs = random.sample(docs, self.config.group_size - 1)          

            inner_batch_data = []
            inner_batch_data.append(self.create_one_example(query, pos))
            for neg in negs:
                inner_batch_data.append(self.create_one_example(query, neg))

            self.samples.append({
                "inputs":inner_batch_data
                })

    def create_one_example(self, qry_text: str, doc_text: str):
        qry_tokenized = self.tokenize_query(qry_text)
        doc_tokenized = self.tokenize_doc(doc_text)
        tokenized = {k : qry_tokenized[k][:-2] + doc_tokenized[2:] for k in  qry_tokenized}

        padding_length = max(self.config.max_len - len(tokenized['input_ids']), 0)
        tokenized['input_ids'] += padding_length*[self.tokenizer.pad_token_id]
        tokenized['attention_mask'] += padding_length*[0]
        if 'token_type_ids' in tokenized:
            tokenized['token_type_ids'] += padding_length*[0]
        return {
            k: torch.tensor(tokenized[k], dtype=torch.long).unsqueeze(0) for k in tokenized
        }


    def tokenize_query(self, query:str):
        query_hash = hash(query)
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]
        else:
            tokenized_query = self.tokenize_pair(
                query, '', max_len= self.config.max_len*3//4
             )
            self.query_cache[query_hash] = tokenized_query
            return tokenized_query

    def tokenize_doc(self, doc:str):
        doc_hash = hash(doc)
        if doc_hash in self.doc_cache:
            return self.doc_cache[doc_hash]
        else:
            tokenized_doc = self.tokenize_pair(
                '', doc, max_len= self.config.max_len*1//4
             )
            self.doc_cache[doc_hash] = tokenized_doc
            return tokenized_doc
    
    def tokenize_pair(self, qry_text: str, doc_text: str, max_len:int):
        item = self.tokenizer.encode_plus(
            qry_text,
            doc_text,
            truncation=True,
            max_length=max_len,
            return_tensors = None,
            # padding='max_length',
        )
        return item

# ========================
# V1 使用
# ========================

# class BgeRerankerEvalDataset(BgeRerankerDataset):
#     '''
#     input json format:
#     {   
#         "query": "query",
#         "pos": ["pos"],
#         "docs": ["doc1","doc2","doc3"..],
#         "pos_mask": [1,0,0,...]
#     }
#     '''
#     def create_samples(self):

#         self.query_cache = {}
#         self.doc_cache   = {}

#         for item in self.data:
#             query = item['query']
#             if isinstance(query, list):
#                 query = query[0]
#             if self.config.query_instruction_for_retrieval is not None:
#                 query = self.config.query_instruction_for_retrieval +  query
#             if self.config.contain_inner_neg:
#                 inner_neg_end = item['inner_neg_end']
#             else:
#                 inner_neg_end = 0

#             docs        = item['docs'][inner_neg_end:self.config.group_size+inner_neg_end]
#             pos_mask    = item['pos_mask'][inner_neg_end:self.config.group_size+inner_neg_end]
#             inner_batch_data = []
#             for neg in docs:
#                 inner_batch_data.append(self.create_one_example(query, neg))

#             self.samples.append({
#                 "inputs":inner_batch_data,
#                 "pos_mask":torch.tensor( pos_mask, dtype = torch.long)
#                 })

# ========================
# V2 使用
# ========================

class BgeRerankerEvalDataset(BgeRerankerDataset):
    '''
    input json format:
    {   
        "query": "query",
        "pos": ["pos"],
        "docs": ["doc1","doc2","doc3"..],
        "pos_mask": [1,0,0,...]
    }
    '''

    def create_samples(self):

        self.query_cache = {}
        self.doc_cache   = {}

        for item in self.data:
            query = item['query']
            if isinstance(query, list):
                query = query[0]
            if self.config.query_instruction_for_retrieval is not None:
                query = self.config.query_instruction_for_retrieval +  query

            docs        = item['docs']
            pos_mask    = item['pos_mask']
            if 1 in pos_mask:
                pos_index = pos_mask.index(1)
            else:
                pos_index = None

            if pos_index is not None:
                pos = docs.pop(pos_index)
            else:
                pos = item['pos'][0]

            if self.config.contain_inner_neg:
                inner_neg_end = item['inner_neg_end']
            else:
                inner_neg_end = 0
            inner_neg = docs[:inner_neg_end]
            docs = inner_neg + docs[inner_neg_end:self.config.group_size-1]

            # if len(docs) < self.config.group_size - 1:
            #     num = math.ceil((self.config.group_size - 1) / len(item['neg']))
            #     negs = random.sample(docs * num, self.config.group_size - 1)
            # else:
            #     negs = random.sample(docs, self.config.group_size - 1)          

            inner_batch_data = []
            inner_batch_data.append(self.create_one_example(query, pos))
            for neg in docs:
                inner_batch_data.append(self.create_one_example(query, neg))
            is_pos_mask = [0]*self.config.group_size
            is_pos_mask[inner_neg_end] = 1
            self.samples.append({
                "inputs":inner_batch_data,
                "pos_mask":torch.tensor( is_pos_mask, dtype = torch.long)
                })