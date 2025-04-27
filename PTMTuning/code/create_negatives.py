import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import faiss
from transformers import AutoTokenizer, AutoModel
import re
# for surport gpu faiss
# from faiss import StandardGpuResources

# 设置常量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PER_GPU_BATCH_SIZE = 32

# 加载模型
MODEL_NAME = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model_config = {"normalize": True}  # BGE模型需要归一化

# 编码函数
def encode_texts(texts):
    all_vectors = []
    for i in tqdm(range(0, len(texts), PER_GPU_BATCH_SIZE)):
        batch_texts = texts[i:i+PER_GPU_BATCH_SIZE]
        
        # 准备输入
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        # 提取嵌入向量
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # 使用CLS token
    
        # 归一化（如果需要）
        if model_config.get("normalize", False):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_vectors.append(embeddings.cpu().to(torch.float32).numpy())
    
    return np.concatenate(all_vectors, axis=0)

# 查询格式化函数
def formatting_func(query):
    task_description = """Retrieve the key misconception behind the wrong answer when given a math problem and its incorrect and correct solutions."""
    return f"Instruct: {task_description}\nQuery: {query}"

def get_query(row):
    query = ""
    query = f"{row['SubjectName']} - {row['ConstructName']}\n" if 'SubjectName' in row and 'ConstructName' in row else ""
    query += f"# Question: {row['QuestionText']}\n"
    query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
    query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
    query = formatting_func(query)
    return query

# 预处理函数：解析query_id得到original_query_id
def parse_query_id(query_id):
    match = re.match(r"(.+)_([A-D])", query_id)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError(f"Invalid query_id format: {query_id}")

# 预处理函数：构建缓存和索引
def build_indices_and_caches(query_df, content_pool_df):
    """
    预处理数据，构建索引和缓存
    
    参数:
    - query_df: 查询数据DataFrame
    - content_pool_df: 内容池DataFrame
    
    返回:
    - 索引和缓存的字典
    """
    print("开始构建索引和缓存...")
    
    # 1. 添加original_query_id列到query_df
    query_df['original_query_id'] = query_df['query_id'].apply(lambda x: parse_query_id(x)[0])
    # res = StandardGpuResources()


    # 2. 构建最难负样本的缓存
    hardest_negatives_cache = {}
    for _, row in tqdm(query_df.iterrows(), desc="构建最难负样本缓存"):
        hardest_negatives_cache[row['query_id']] = query_df[
            (query_df['original_query_id'] == row['original_query_id']) &
            (query_df['content_id'] != row['content_id'])
        ]['content_id'].unique().tolist()
    
    # 3. 为内容池创建嵌入和索引
    print("为内容池创建嵌入...")
    
    content_texts = content_pool_df['MisconceptionName'].tolist()
    content_ids = content_pool_df['content_id'].tolist()
    content_id_to_idx = {cid: idx for idx, cid in enumerate(content_ids)}
    content_embeddings = encode_texts(content_texts)
    
    # 创建faiss索引
    # 创建内容池faiss索引（GPU版本）
    # print("创建内容池faiss索引...")
    # dim = content_embeddings.shape[1]
    # # 将索引转移到GPU
    # cpu_index = faiss.IndexFlatIP(dim)  # 先创建CPU索引

    # content_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 使用第一个GPU
    # content_index.add(content_embeddings.astype('float32'))

    print("创建内容池faiss索引...")
    dim = content_embeddings.shape[1]
    content_index = faiss.IndexFlatIP(dim)
    content_index.add(content_embeddings)
    
    # 4. 构建题目相似度索引
    # 构建original_query_id到内容的映射
    print("构建题目映射...")
    oriqid_based_content_dict = {}
    for _, row in query_df.iterrows():
        oriq_id = row['original_query_id']
        if oriq_id not in oriqid_based_content_dict:
            oriqid_based_content_dict[oriq_id] = {
                'QuestionText': row['QuestionText'],
                'content_ids': []
            }
        if row['content_id'] not in oriqid_based_content_dict[oriq_id]['content_ids']:
            oriqid_based_content_dict[oriq_id]['content_ids'].append(row['content_id'])
    
    # 创建题目嵌入
    print("为题目创建嵌入...")
    unique_oriq_ids = list(oriqid_based_content_dict.keys())
    question_texts = [oriqid_based_content_dict[oriq_id]['QuestionText'] for oriq_id in unique_oriq_ids]
    question_embeddings = encode_texts(question_texts)
    
    # 创建题目相似度索引
    print("创建题目faiss索引...")
    question_index = faiss.IndexFlatIP(dim)
    question_index.add(question_embeddings)

    # 创建题目相似度索引（GPU版本）
    # print("创建题目faiss索引...")
    # cpu_question_index = faiss.IndexFlatIP(dim)
    # question_index = faiss.index_cpu_to_gpu(res, 0, cpu_question_index)
    # question_index.add(question_embeddings.astype('float32'))


    # 5. 为查询创建嵌入
    print("为查询创建嵌入...")
    query_df['alltext'] = query_df.apply(
        lambda row: get_query({
            'QuestionText': row['QuestionText'],
            'CorrectAnswerText': row['CorrectAnswerText'],
            'InCorrectAnswerText': row['InCorrectAnswerText'],
            'SubjectName': row.get('SubjectName', ''),
            'ConstructName': row.get('ConstructName', '')
        }), axis=1
    )
    
    query_texts = query_df['alltext'].tolist()
    query_embeddings = encode_texts(query_texts)
    query_id_to_embedding = {row['query_id']: query_embeddings[i] for i, (_, row) in enumerate(query_df.iterrows())}
    
    # 收集并返回所有缓存和索引
    cache_dict = {
        'hardest_negatives_cache': hardest_negatives_cache,
        'question_embeddings': question_embeddings.astype('float32'),
        'content_index': content_index,
        'content_embeddings': content_embeddings.astype('float32'),
        'content_ids': content_ids,
        'content_id_to_idx': content_id_to_idx,
        'question_index': question_index,
        'unique_oriq_ids': unique_oriq_ids,
        'oriqid_based_content_dict': oriqid_based_content_dict,
        'query_id_to_embedding': query_id_to_embedding
    }
    
    print("索引和缓存构建完成")
    return cache_dict

# 改进后的难负样本生成函数
def generate_hard_negatives(query_row, cache_dict, topk=10, content_sim_threshold=0.7, query_sim_threshold=0.8):
    """
    为给定的查询生成难负样本
    
    参数:
    - query_df: 查询数据DataFrame
    - query_row: 当前查询行
    - cache_dict: 索引和缓存字典
    - topk: 总共需要生成的难负样本数量
    - content_sim_threshold: 内容相似度阈值
    - query_sim_threshold: 查询相似度阈值
    
    返回:
    - hard_negatives: 难负样本content_id列表
    - semi_negatives: semi难负样本content_id列表
    """
    
    hard_negatives = []
    semi_negatives = []

    current_query_id = query_row['query_id']
    current_content_id = query_row['content_id']
    original_query_id = query_row['original_query_id']
    
    # 步骤1: 获取最难负样本 - 使用缓存直接获取
    hardest_negatives = cache_dict['hardest_negatives_cache'].get(current_query_id, [])
    hard_negatives.extend(hardest_negatives)
    
    # 步骤2: 基于content相似度获取负样本
    # 获取当前内容的嵌入
    current_content_idx = cache_dict['content_id_to_idx'].get(current_content_id)
    if current_content_idx is not None:
        current_content_embedding = cache_dict['content_embeddings'][current_content_idx].reshape(1, -1)
        
        # 搜索相似内容
        sim_scores, sim_indices = cache_dict['content_index'].search(current_content_embedding, 50)
        
        content_similar_negatives = []
        for i, idx in enumerate(sim_indices[0]):
            if sim_scores[0][i] > content_sim_threshold:
                content_id = cache_dict['content_ids'][idx]
                if content_id != current_content_id and content_id not in hard_negatives:
                    content_similar_negatives.append(content_id)
                    if len(content_similar_negatives) >= 3:
                        break
        
        hard_negatives.extend(content_similar_negatives)
    
    # 步骤3: 基于query相似度获取负样本
    # 获取当前题目的嵌入
    try:
        oriq_idx = cache_dict['unique_oriq_ids'].index(original_query_id)
        # current_question_embedding = cache_dict['question_index'].reconstruct(oriq_idx).reshape(1, -1)

        current_question_embedding = cache_dict['question_embeddings'][oriq_idx].reshape(1, -1)
        
        # 搜索相似题目
        q_sim_scores, q_sim_indices = cache_dict['question_index'].search(current_question_embedding, 50)
        
        query_similar_negatives = []
        for i, idx in enumerate(q_sim_indices[0]):
            if q_sim_scores[0][i] > query_sim_threshold and idx != oriq_idx:
                oriq_id = cache_dict['unique_oriq_ids'][idx]
                # 从这个题目中选择内容ID
                available_content_ids = [cid for cid in cache_dict['oriqid_based_content_dict'][oriq_id]['content_ids'] 
                                        if cid != current_content_id and cid not in hard_negatives]
                
                if available_content_ids:
                    # 添加不超过2个
                    for cid in available_content_ids[:min(2, 2-(len(query_similar_negatives)))]:
                        query_similar_negatives.append(cid)
                        if len(query_similar_negatives) >= 2:
                            break
            
            if len(query_similar_negatives) >= 2:
                break
        
        hard_negatives.extend(query_similar_negatives)
    except (ValueError, KeyError) as e:
        print(f"处理查询相似度时出错: {str(e)}")
    
    # 步骤4: 基于query检索content补足剩余所需负样本
    remaining_count = topk - len(hard_negatives)

    if current_query_id in cache_dict['query_id_to_embedding']:
        query_embedding = cache_dict['query_id_to_embedding'][current_query_id].reshape(1, -1)
        _, q_content_indices = cache_dict['content_index'].search(query_embedding, 300)

        semi_content_negatives = []
        for i, idx in enumerate(q_content_indices[0][100:300]):
            cid = cache_dict['content_ids'][idx]
            if cid != current_content_id and cid not in semi_negatives:
                semi_content_negatives.append(cid)
                if len(semi_content_negatives) >= topk:
                    break
        
        if remaining_count > 0 :
            query_content_negatives = []
            for i, idx in enumerate(q_content_indices[0][:100] ):
                cid = cache_dict['content_ids'][idx]
                if cid != current_content_id and cid not in hard_negatives:
                    query_content_negatives.append(cid)
                    if len(query_content_negatives) >= remaining_count:
                        break
        
        hard_negatives.extend(query_content_negatives)
        semi_negatives.extend(semi_content_negatives)     

    
    return hard_negatives,semi_negatives

# 改进后的主处理函数
def process_all_queries(query_df, content_pool_df, sample_count=None):
    """
    处理所有查询并生成难负样本和semi难负样本
    
    参数:
    - query_df: 查询数据DataFrame
    - content_pool_df: 内容池DataFrame
    - sample_count: 处理的样本数量，用于测试
    
    返回:
    - results_df_hard: 包含查询ID和对应难负样本的DataFrame
    - results_df_semi: 包含查询ID和对应semi负样本的DataFrame
    """
    # 构建所有必要的索引和缓存
    cache_dict = build_indices_and_caches(query_df, content_pool_df)
    
    results_hard = []
    results_semi = []
    
    # 如果指定了样本数量，则只处理部分数据
    if sample_count:
        query_sample = query_df.sample(sample_count)
    else:
        query_sample = query_df
    
    for idx, row in tqdm(query_sample.iterrows(), total=len(query_sample), desc="Processing queries"):
        print(f"\n处理查询 {idx+1}/{len(query_sample)}: {row['query_id']}")
        try:
            # 生成hard负样本
            hard_negatives,semi_negatives = generate_hard_negatives(row, cache_dict)
            results_hard.append({
                'query_id': row['query_id'],
                'content_id': row['content_id'],  # 正样本
                'hard_negatives': ",".join([ str(i) for i in  hard_negatives] ),  # 难负样本
                'hard_negative_count': len(hard_negatives)
            })
            print(f"为查询 {row['query_id']} 生成了 {len(hard_negatives)} 个hard负样本")
    
            # 生成semi负样本
            results_semi.append({
                'query_id': row['query_id'],
                'content_id': row['content_id'],  # 正样本
                'semi_negatives': ",".join([ str(i) for i in semi_negatives] ),  # semi负样本
                'semi_negative_count': len(semi_negatives)
            })
            print(f"为查询 {row['query_id']} 生成了 {len(semi_negatives)} 个semi负样本")
            
        except Exception as e:
            print(f"处理查询 {row['query_id']} 时出错: {str(e)}")
    
    # 转换为DataFrame
    results_df_hard = pd.DataFrame(results_hard)
    results_df_semi = pd.DataFrame(results_semi)
    return results_df_hard, results_df_semi

# 使用示例
def main():
    # 加载数据
    # 假设这些数据已经存在
    query_df = pd.read_csv('../input/synthetic.csv')
    content_pool_df = pd.read_csv('../input/eedi_content.csv')
    content_pool_df = content_pool_df.rename(columns={"MisconceptionId":"content_id"})
    
    # 处理一小部分查询进行测试
    # print("处理小样本测试...")
    # sample_results = process_all_queries(query_df, content_pool_df, sample_count=5)
    
    # # 保存结果
    # sample_results.to_csv('hard_negatives_sample.csv', index=False)
    
    # 处理所有查询
    print("\n处理所有查询...")
    full_results_hard, full_results_semi = process_all_queries(query_df, content_pool_df)

    full_results_hard.to_csv('negatives_hard.csv', index=False)
    full_results_semi.to_csv('negatives_semi.csv', index=False)  # 按要求保存semi负样本
    
    print("难负样本和semi负样本生成完成！")
    print("难负样本生成完成！")

if __name__ == "__main__":
    main()