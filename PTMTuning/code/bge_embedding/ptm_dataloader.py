import torch


class TripletCollator:
    """
    Collator for triplet data with semi-hard and hard negative samples
    """
    def __call__(self, batch):
        batch_out = {}
        for sample in batch:
            for key, value in sample.items():
                if key == 'query_id':
                    batch_out.setdefault(key, []).append(value)
                else:
                    for sub_key, sub_value in value.items():
                        batch_out.setdefault(key, {}).setdefault(sub_key, []).append(sub_value)

        for key, value in batch_out.items():
            if key == 'query_id':
                batch_out[key] = value  # 保持字符串
            else:
                for sub_key, sub_value in value.items():
                    batch_out[key][sub_key] = torch.cat(sub_value, dim=0) if sub_value[0].dim() > 1 else torch.stack(sub_value, dim=0)
                batch_out[key] = truncate_to_max_length(batch_out[key])##长度截断
                    
        
        return batch_out
    
class TextCollator:
    """
    Collator for triplet data with semi-hard and hard negative samples
    """

    def __call__(self, batch):
        result = {
            'input_ids': [],
            'attention_mask': []
        }
        
        # 如果有ID字段则收集
        if 'id' in batch[0]:
            result['id'] = []
        
        # 收集每个样本的字段
        for sample in batch:
            for key in result:
                if key in sample:
                    result[key].append(sample[key])
        
        # 将列表转换为张量，仅对张量类型的字段
        for key in result:
            if key != 'id' and result[key]:
                if isinstance(result[key][0], torch.Tensor):
                    result[key] = torch.stack(result[key], dim=0)
        
        return truncate_to_max_length(result)


def truncate_to_max_length(input_dict):
    """截断所有字段到attention_mask的最大有效长度"""
    if 'attention_mask' not in input_dict:
        return input_dict
    
    # 计算最大有效长度
    mask = input_dict['attention_mask']
    max_length = mask.sum(dim=-1).max().item()
    
    # 截断所有字段的最后一维
    truncated = {k: v[..., :max_length] for k, v in input_dict.items()}
    return truncated



# ---
def show_batch(batch, tokenizer, n_examples=4, print_fn=print):
    bs = batch["query"]["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids (query): {batch['query']['input_ids'].shape}")
    print_fn(f"shape of input_ids (content): {batch['contents']['input_ids'].shape}")
    print_fn(f"shape of input_ids (cot): {batch['cot']['input_ids'].shape}")

    print("--" * 80)
    for idx in range(n_examples):
        print_fn(f"[Query]:\n{tokenizer.decode(batch['query']['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 20)
        print_fn(f"[Content]:\n{tokenizer.decode(batch['contents']['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 20)
        print_fn(f"[COT]:\n{tokenizer.decode(batch['cot']['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("--" * 80)


def show_batch_fs(batch, tokenizer, n_examples=4, print_fn=print):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    print("--" * 80)
    for idx in range(n_examples):
        print_fn(f"[Query]:\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 20)