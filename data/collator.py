# This is the collator.py file in data folder
from transformers    import DataCollatorWithPadding
from dataclasses     import dataclass
import torch

@dataclass
class BgeEmbedCollator(DataCollatorWithPadding):
    def __call__(self, batched_item):
        batched_collate = self.recursive_collate(batched_item)
        for key in batched_collate:
            if not isinstance(batched_collate[key], dict):
                continue
            if 'attention_mask' in batched_collate[key]:
                max_length_in_batch = batched_collate[key]['attention_mask'].sum(dim=1).max().item()
                for subkey in batched_collate[key]:
                    batched_collate[key][subkey] = batched_collate[key][subkey][:, :max_length_in_batch]
        return batched_collate
    
    def recursive_collate(self, list_of_M):
        """
        将 list 中的每个嵌套字典中的 torch.Tensor 沿着第一维度拼接，并保持嵌套结构不变。
        
        参数：
        list_of_M: 一个嵌套字典的 list，最底层是 torch.Tensor，所有字典结构相同
        
        返回：
        一个拼接后的嵌套字典
        """
        # 获取第一个字典作为结构模板
        first_M = list_of_M[0]
        
        # 如果是字典，递归调用函数处理每个子项
        if isinstance(first_M, dict):
            collated_dict = {}
            for key in first_M:
                # 对每个key递归处理
                sub_list_of_M = [m[key] for m in list_of_M]  # 收集当前key的所有子字典/子tensor
                collated_dict[key] = self.recursive_collate(sub_list_of_M)
            return collated_dict

        # 如果是torch.Tensor，沿着第一维度进行拼接
        elif isinstance(first_M, torch.Tensor):
            return torch.cat(list_of_M, dim=0)

        # 如果不是dict或者tensor，则不处理（可以根据需求自定义）
        else:
            raise ValueError("Unsupported data type encountered in the dictionary.")
        


def reranker_collate(features):
    # inputs 是 list of dict
    # 将 pos_mask 进行拼接
    output = {}
    if 'pos_mask' in features[0]:
        output['pos_mask'] = torch.stack([feature['pos_mask'] for feature in features], dim=0)

    inputs = sum([ feature['inputs'] for feature in features], [])
    output['inputs'] = {
        k:torch.cat([feature[k] for feature in inputs], dim=0)
          for k in inputs[0].keys()
    }
    del inputs
    _max_length = output['inputs']['attention_mask'].sum(dim=1).max().item()
    for k in output['inputs'].keys():
        output['inputs'][k] = output['inputs'][k][:, :_max_length]
    return output



'''
现在我有两个dataframe，一个query相关的，一个content相关的
query相关：query_id 唯一表示，query_text（这个是虚拟的，我代码中会实现生成）,content_id,content,semi_hard_content_Ids,hard_content_Ids
content_data 相关: content_id, content 这里是完整的content_id到content的映射，query中只有一个对应一个pos content

第一种：为了生成inbatch的情况，使用字段 query的数据 query_text, content 
    - query生成query_text的dataset
    - query的content生成 content的dataset
第二种：为了生成 multi neg 的情况，使用字段 query的数据 query_text, content ，而content相关使用content
    - query生成query_text的query dataset，同时保留 query_id到 semi_hard_content_Ids/hard_content_Ids的映射关系 query_to_content 。例如semi_hard_content_Ids为"22,333,1",query_id为7，则生成{7:[22,333,1]}
    - query的content生成 content的 contents dataset，其中保留content_id
    - 创建一个新的dataset，输入是 query dataset 和 contents dataset，以 query dataset 中query_id作为索引，根据 query_to_content 的 当下query的contents 去 contents dataset 获取其对应的 tokenzied

请问第二种的实现方法是否合理，下面是我的代码实现

'''