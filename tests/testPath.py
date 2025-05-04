from ..config.configs import  RetrieverModelConfig
from ..utils.utilities import ProjPathsFiles
def testRetrieverModelConfig():
    args_dict = {'model_name': 'nice',
    'model_path': "resource/bge_model_pt",
    'output_dir':"resource/bge_model_ft",
    'use_inbatch_neg': True,
    'temperature': 1 }
    print("test,.....")
    trLoaderConfig = RetrieverModelConfig.from_dict(args_dict)
    print(trLoaderConfig)

    print(ProjPathsFiles(args_dict['model_path']).abs_path_or_file)
    return trLoaderConfig

if __name__ == '__main__':
    print(1233444)
    testRetrieverModelConfig()

根据后面提供的一些信息，编写代码得到"期望打印的内容"，其中召回使用faiss。（注意：可以先召回较多的内容，然后根据需要topk计算相关内容）


期望打印内容伪代码：

```python
cutoffs = [1, 2, 4, 8, 16, 25, 32, 64]
print("--------------------------------")
print(f">>> LB: {scores_dict['lb']}")
print(f">>> Seen LB: {scores_dict['seen_lb']}")
print(f">>> Unseen LB: {scores_dict['unseen_lb']}")
print("--------------------------------")

for pt in cutoffs:
    print(f">>> Current Recall@{pt} = {round(scores_dict[f'recall@{pt}'], 4)}")

```

输入：
  querydf{ 'SubjectName', 'ConstructName','QuestionText','CorrectAnswerText','InCorrectAnswerText','content_id'}
  contentdf{ 'MisconceptionId', 'MisconceptionName' }#MisconceptionId 和 content_id是一样的


# 以下函数用于 query的文本生成

def _formatting_func(query):
    task_description = """Retrieve the key misconception behind the wrong answer when given a math problem and its incorrect and correct solutions."""

    return f"Instruct: {task_description}\nQuery: {query}"


def _get_query(row):
    query = ""
    query = f"{row['SubjectName']} - {row['ConstructName']}\n"
    query += f"# Question: {row['QuestionText']}\n"
    query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
    query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
    query = _formatting_func(query)
    return query


模型初始化逻辑：
```python
backbone_path = 'BAAI/bge-large-en-v1.5'
config = AutoConfig.from_pretrained(
            backbone_path,
            trust_remote_code=False
        )
config.use_cache = False
model_kwargs = {
        "config": config,
        "trust_remote_code": False,
        "attn_implementation": 'sdpa'
    }

base_model = AutoModel.from_pretrained(backbone_path, **model_kwargs)
```

文本encode逻辑：

```python
    def sentence_embedding(self, hidden_state, mask):
        return hidden_state[:, 0]

    def encode(self, features, mode):
        if self.sub_batch_size is not None and self.sub_batch_size > 0:
            all_p_reps = []
            for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                #memory usage care
                end_inx = min(i + self.sub_batch_size, len(features["attention_mask"]))
                sub_features = {k: v[i:end_inx] for k, v in features.items()}
                last_hidden_state = self.backbone(**sub_features, return_dict=True).last_hidden_state
                p_reps = self.sentence_embedding(last_hidden_state, sub_features["attention_mask"])
                all_p_reps.append(p_reps)
                del p_reps, last_hidden_state, sub_features
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
        else:
            last_hidden_state = self.backbone(**features, return_dict=True).last_hidden_state
            all_p_reps = self.sentence_embedding(last_hidden_state, features["attention_mask"])

        return all_p_reps


```


score相关代码：
```python

import numpy as np


def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """

    return round(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]), 4)


def _compute_metrics(true_ids, pred_ids, debug=False):
    """
    fbeta score for one example
    """

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # calculate the confusion matrix variables
    tp = len(true_ids.intersection(pred_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    # metrics
    f1 = tp / (tp + 0.5 * fp + 0.5 * fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if debug:
        print("Ground truth count:", len(true_ids))
        print("Predicted count:", len(pred_ids))
        print("True positives:", tp)
        print("False positives:", fp)
        print("F2:", f2)

    to_return = {
        "f1": f1,
        "f2": f2,
        "precision": precision,
        "recall": recall,
    }

    return to_return


def compute_retrieval_metrics(true_ids, pred_ids):
    """
    fbeta metric for learning equality - content recommendation task

    :param true_ids: ground truth content ids
    :type true_ids: List[List[str]]
    :param pred_ids: prediction content ids
    :type pred_ids: List[List[str]]
    """
    assert len(true_ids) == len(pred_ids), "length mismatch between truths and predictions"
    n_examples = len(true_ids)
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(n_examples):
        ex_true_ids = true_ids[i]
        ex_pred_ids = pred_ids[i]
        if len(ex_pred_ids) == 0:
            f1 = 0.0
            f2 = 0.0
            precision = 0.0
            recall = 0.0
        else:
            m = _compute_metrics(ex_true_ids, ex_pred_ids)
            f1 = m["f1"]
            f2 = m["f2"]
            precision = m["precision"]
            recall = m["recall"]

        f1_scores.append(f1)
        f2_scores.append(f2)
        precision_scores.append(precision)
        recall_scores.append(recall)

    to_return = {
        "f1_score": np.mean(f1_scores),
        "f2_score": np.mean(f2_scores),
        "precision_score": np.mean(precision_scores),
        "recall_score": np.mean(recall_scores),
    }

    return to_return

```




下面是我之前设置的学习率情况，以及运行的效果，根据这个效果，你分析下，你刚刚给出的学习率是否需要调整：

---------------以下是运行效果------------------


### qcot train 30 epoch low lr
train_params:
  warmup_pct: 0.08 # 0.15-->0.05或0.08  
  num_epochs: 30 # 10-->30  

optimizer
  lr:  1e-5
  lr_lora_a:  1e-5
  lr_lora_b: 5e-5
  lr_embed_tokens: 6e-5
  lr_head: 4e-5 


#### epoch 0
>>> LB: 0.1645                                                                                               
--------------------------------
>>> LB: 0.1645
>>> Seen LB: 0.1662
>>> Unseen LB: 0.1441
--------------------------------
>>> Current Recall@1 = 0.0737
>>> Current Recall@2 = 0.129
>>> Current Recall@4 = 0.2295
>>> Current Recall@8 = 0.3585
>>> Current Recall@16 = 0.4673
>>> Current Recall@25 = 0.531
>>> Current Recall@32 = 0.5796
>>> Current Recall@64 = 0.6616
#### epoch 1
>>> LB: 0.1934
>>> Seen LB: 0.1964
>>> Unseen LB: 0.1565
--------------------------------
>>> Current Recall@1 = 0.0955
>>> Current Recall@2 = 0.1575
>>> Current Recall@4 = 0.2596
>>> Current Recall@8 = 0.397
>>> Current Recall@16 = 0.5209
>>> Current Recall@25 = 0.5829
>>> Current Recall@32 = 0.6198
>>> Current Recall@64 = 0.7052

#### epoch 2

>>> LB: 0.2096                                                                                                            
--------------------------------
>>> LB: 0.2096
>>> Seen LB: 0.214
>>> Unseen LB: 0.154
--------------------------------
>>> Current Recall@1 = 0.1005
>>> Current Recall@2 = 0.1977
>>> Current Recall@4 = 0.2898
>>> Current Recall@8 = 0.4221
>>> Current Recall@16 = 0.541
>>> Current Recall@25 = 0.5963
>>> Current Recall@32 = 0.6348
>>> Current Recall@64 = 0.7136


#### epoch 3
>>> LB: 0.1933
>>> Seen LB: 0.1972
>>> Unseen LB: 0.144
--------------------------------
>>> Current Recall@1 = 0.0988
>>> Current Recall@2 = 0.1776
>>> Current Recall@4 = 0.2529
>>> Current Recall@8 = 0.3652
>>> Current Recall@16 = 0.5025
>>> Current Recall@25 = 0.5578
>>> Current Recall@32 = 0.603
>>> Current Recall@64 = 0.6851
#### epoch 4
>>> LB: 0.1392                                                                                                                
--------------------------------
>>> LB: 0.1392
>>> Seen LB: 0.1416
>>> Unseen LB: 0.1087
--------------------------------
>>> Current Recall@1 = 0.0704
>>> Current Recall@2 = 0.1122
>>> Current Recall@4 = 0.1792
>>> Current Recall@8 = 0.2747
>>> Current Recall@16 = 0.3869
>>> Current Recall@25 = 0.4472
>>> Current Recall@32 = 0.4908
>>> Current Recall@64 = 0.598
#### epoch 5