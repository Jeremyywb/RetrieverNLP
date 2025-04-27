# 为提升效率进行优化，尤其针对tokenizer共享、样本复用与数据结构设计

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from transformers import AutoTokenizer
from typing import Dict, List


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path, add_eos_token=cfg.model.add_eos_token)
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = cfg.model.padding_side  # "left"
    return tokenizer


def _formatting_func(query):
    task_description = """Retrieve the key misconception behind the wrong answer when given a math problem and its incorrect and correct solutions."""

    return f"Instruct: {task_description}\nQuery: {query}"


class BaseDataset(Dataset):
    def __init__(self, dataframe, id_column, text_column, tokenizer, max_length=128):
        self.df = dataframe
        self.id_column = id_column
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ids = self.df[id_column].astype(str).tolist()
        self.id_to_idx = {id_val: idx for idx, id_val in enumerate(self.ids)}
        if text_column == 'QuestionText':
            self.df['QuestionText'] = self.df.apply(self._get_query, axis=1)
        # 提前批量tokenize全部文本以加速
        self.encoded_texts = self.tokenizer(
            self.df[text_column].astype(str).tolist(),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def _get_query(self, row):
        query = ""
        query = f"{row['SubjectName']} - {row['ConstructName']}\n"
        query += f"# Question: {row['QuestionText']}\n"
        query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
        query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
        query = _formatting_func(query)
        return query
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            # 'id': self.ids[idx],
            'input_ids': self.encoded_texts['input_ids'][idx],
            'attention_mask': self.encoded_texts['attention_mask'][idx]
        }

    def get_by_id(self, id_val):
        idx = self.id_to_idx.get(str(id_val), None)
        return self[idx] if idx is not None else None


class QueryDataset(BaseDataset):
    def __init__(self, df, tokenizer, max_length=128):
        super().__init__(df, 'query_id', 'QuestionText', tokenizer, max_length)
        self.query_to_content = dict(zip(df['query_id'].astype(str), df['content_id'].astype(str)))

    def get_content_id(self, query_id):
        return self.query_to_content.get(str(query_id))


class ContentDataset(BaseDataset):
    def __init__(self, df, tokenizer, max_length=128):
        super().__init__(df, 'content_id', 'MisconceptionName', tokenizer, max_length)


class CotDataset(BaseDataset):
    def __init__(self, df, tokenizer, max_length=128):
        super().__init__(df, 'query_id', 'Explanation', tokenizer, max_length)

class RetrieverDataset(Dataset):
    def __init__(
        self,
        cfg,
        query_dataset: QueryDataset,
        content_dataset: ContentDataset,
        cot_dataset: CotDataset,
        external_cot_dataset: CotDataset,
        negatives: Dict[str, List[str]] = None
    ):
        self.query_dataset = query_dataset
        self.content_dataset = content_dataset
        self.cot_dataset = cot_dataset
        self.external_cot_dataset = external_cot_dataset
        self.mode = cfg.mode

        self.query_ids = self.query_dataset.ids
        self.query_to_semi_content_ids = negatives if cfg.mode == 'semi' else {}
        self.query_to_hard_content_ids = negatives if cfg.mode == 'hard' else {}

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]

        query_item = self.query_dataset.get_by_id(query_id)
        pos_content_id = self.query_dataset.get_content_id(query_id)
        pos_item = self.content_dataset.get_by_id(pos_content_id)

        # 获取负样本
        if self.mode == 'semi':
            neg_ids = self.query_to_semi_content_ids.get(query_id, [])
        elif self.mode == 'hard':
            neg_ids = self.query_to_hard_content_ids.get(query_id, [])
        else:
            neg_ids = []

        neg_items = [
            self.content_dataset.get_by_id(cid)
            for cid in neg_ids
            if self.content_dataset.get_by_id(cid) is not None
        ]

        all_content_input_ids = [pos_item['input_ids']] + [n['input_ids'] for n in neg_items]
        all_content_attention = [pos_item['attention_mask']] + [n['attention_mask'] for n in neg_items]

        # 获取 COT + External COT
        cot_item = self.cot_dataset.get_by_id(query_id)
        ext_cot_item = self.external_cot_dataset.get_by_id(query_id)

        cot_input_ids = torch.stack([cot_item['input_ids'], ext_cot_item['input_ids']], dim=0)
        cot_attention_mask = torch.stack([cot_item['attention_mask'], ext_cot_item['attention_mask']], dim=0)

        return {
            # 'query_id': query_id,
            'query': {
                'input_ids': query_item['input_ids'],
                'attention_mask': query_item['attention_mask']
            },
            'contents': {
                'input_ids': torch.stack(all_content_input_ids, dim=0),
                'attention_mask': torch.stack(all_content_attention, dim=0)
            },
            'cot': {
                'input_ids': cot_input_ids,
                'attention_mask': cot_attention_mask
            }
        }

