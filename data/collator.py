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