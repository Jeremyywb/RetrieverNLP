from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from abc import ABC, abstractmethod

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


class MathDataset(ABC):
    """
    Dataset class for processing EEDI math MCQs into query/content inputs for retrieval
    # input
    - cfg: for init input,configuration of dataset
    - df : for preprocess,query dataset with content ids
    """

    def __init__(self, cfg, query_formatting_func=None):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.query_formatting_func = query_formatting_func if query_formatting_func is not None else _formatting_func

    def pre_process(self, df, is_query, is_train=False):
        def _get_query(row):
            query = ""
            query = f"{row['SubjectName']} - {row['ConstructName']}\n"
            query += f"# Question: {row['QuestionText']}\n"
            query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
            query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
            query = self.query_formatting_func(query)
            return query

        def _get_content(row):
            return row["MisconceptionName"]
        if is_query:
            df["text"] = df.apply(lambda x: _get_query(x), axis=1)
            df = df[["query_id", "text"]]
            self.query_id = df['query_id'].values.tolist()
        else:
            df["text"] = df.apply(lambda x: _get_content(x), axis=1)
            df = df.rename(columns={"MisconceptionId": "content_id"})
            df = df[["content_id", "text"]]
            self.content_id = df['content_id'].values.tolist()

        return df


    def tokenize(self, examples, max_len):
        tokenized = self.tokenizer.encode_plus(
                    examples["text"],  # 直接传递文本字符串，而不是放在列表中
                    padding        = 'max_length',
                    return_tensors = 'pt',
                    max_length     = max_len,
                    truncation     = True
                )
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}
    
class MathHardDataset(MathDataset,Dataset):
    """
    Dataset class for processing EEDI math MCQs with pre-selected negative samples
    Uses semi_content_ids and hard_content_ids to get negative samples
    """
    
    def __init__(self, cfg, df, query_formatting_func=None, is_query=True, is_train=False):
        super().__init__(cfg, query_formatting_func)
        # Store query_id to negative ids mapping
        self.is_query = is_query
        self.is_train = is_train
        self.cfg = cfg
        self.df = self.pre_process(df, is_query, is_train)
        self.df = self.df.to_dict(orient='records')
        # is_query: query_id
        # is_train: query_id_to_neg_ids
        # else: content_id 
        
    def pre_process(self, df, is_query, is_train):
        if is_query and is_train:
            self.query_id_to_neg_ids = {} 
            # Store negative ids mapping for queries
            for _, row in df.iterrows():
                query_id = row['query_id']
                pos_content_id = row['content_id']
                self.query_id_to_neg_ids[query_id] = {
                    'semi': [int(pos_content_id)]+[int(id) for id in row['semi_content_ids'].split(',')] if pd.notna(row['semi_content_ids']) else [],
                    'hard': [int(pos_content_id)]+[int(id) for id in row['hard_content_ids'].split(',')] if pd.notna(row['hard_content_ids']) else []
                }
        
        # Call parent's pre_process
        return super().pre_process(df, is_query, is_train)
        
    def __getitem__(self, idx):
        # Get query data
        return self.tokenize(self.df[idx], self.cfg.model.max_length)#(1, max_length)
            
    def __len__(self):
        return len(self.df)


class TripletDataset(Dataset):
    """
    Dataset for triplet training with semi-hard or hard negative samples
    Combines query dataset and content dataset to form triplets
    """
    
    def __init__(self, query_dataset, content_dataset, neg_strategy='semi-hard'):
        self.query_dataset = query_dataset
        self.content_dataset = content_dataset
        self.neg_strategy = neg_strategy
        self.query_id_to_neg_ids = query_dataset.query_id_to_neg_ids
        self.query_id = query_dataset.query_id
        
    def __len__(self):
        return len(self.query_dataset)
        
    def __getitem__(self, idx):
        # Get query data
        query_item = self.query_dataset[idx]
        query_id = self.query_id[idx]
        
        # Get negative content ids
        neg_ids = self.query_id_to_neg_ids[query_id][self.neg_strategy]
        # 百分百确定大于2

        # Get content data
        # First item is positive content


        # Get negative contents
        all_contents = [self.content_dataset[_id] for _id in neg_ids]#num contemt,1,max_length
        
        # Create pos_mask (1 for positive, 0 for negatives)
        pos_mask = [1] + [0] * (len(all_contents)-1)
        
        return {
            'query': query_item,
            'contents': all_contents,
            'pos_mask': pos_mask,
            'query_id': query_id
        }
