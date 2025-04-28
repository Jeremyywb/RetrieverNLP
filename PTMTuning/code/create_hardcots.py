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


# 预处理函数：构建缓存和索引
def build_indices_and_caches(query_df):
    """
    预处理数据，构建索引和缓存
    
    参数:
    - query_df: 查询数据DataFrame
    - content_pool_df: 内容池DataFrame
    
    返回:
    - 索引和缓存的字典
    """
    print("开始构建索引和缓存...")
    query_df = query_df.copy()

    query_df['original_query_id'] = query_df['query_id'].map(lambda x: x.split('_')[0])

    # 2) 构建 orig_to_cot_ids
    orig_to_cot_ids = query_df.groupby('original_query_id')['query_id'].apply(list).to_dict()
    # 3) 构建 query_to_content 映射
    query_to_content = dict(zip(query_df['query_id'], query_df['content_id']))

    # 4) 批量 encode Explanation → embeddings
    cot_texts = query_df['Explanation'].tolist()
    cot_ids   = query_df['query_id'].tolist()

    cot_embeddings = encode_texts(cot_texts)

    # 5) 建 FAISS 索引
    dim = cot_embeddings.shape[1]
    cot_index = faiss.IndexFlatIP(dim)
    cot_index.add(cot_embeddings)

    # 6) id ↔ idx 映射
    cot_id_to_idx = {qid: idx for idx, qid in enumerate(cot_ids)}

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
    


    cache_dict = {
        'question_embeddings': question_embeddings.astype('float32'),
        'question_index': question_index,
        'unique_oriq_ids': unique_oriq_ids,
        'oriqid_based_content_dict': oriqid_based_content_dict,
        'cot_index': cot_index,
        'cot_embeddings': cot_embeddings,
        'cot_ids': cot_ids,
        'cot_id_to_idx': cot_id_to_idx,
        'orig_to_cot_ids': orig_to_cot_ids,
        'query_to_content': query_to_content,
        'query_id_to_embedding': query_id_to_embedding
   
    }
    
    print("索引和缓存构建完成")
    return cache_dict

# 改进后的难负样本生成函数
def generate_cot_hard_negatives(query_row, cache, topk=10,
                                cot_sim_th=0.7, query_sim_th=0.8):
    qid      = query_row['query_id']
    oqid     = query_row['original_query_id']
    pos_cid  = cache['query_to_content'][qid]  # 当前正样本 content_id

    hard = []
    ci   = cache['cot_id_to_idx'][qid]
    emb  = cache['cot_embeddings'][ci:ci+1]

    # Step 2: 基于 CoT 相似度
    sims, idxs = cache['cot_index'].search(emb, 50)
    for s, i in zip(sims[0], idxs[0]):
        cand_qid = cache['cot_ids'][i]
        cand_cid = cache['query_to_content'][cand_qid]
        if s > cot_sim_th and cand_qid != qid and cand_cid != pos_cid:
            hard.append(cand_qid)
            if len(hard) >= 3:
                break


    # Step 3: 基于题目相似度
    oq_list = cache['unique_oriq_ids']
    oq_idx  = oq_list.index(oqid)
    q_emb   = cache['question_embeddings'][oq_idx:oq_idx+1]
    qs, qis = cache['question_index'].search(q_emb, 50)
    cnt = 0
    for score, qi in zip(qs[0], qis[0]):
        if score > query_sim_th and qi != oq_idx:
            sim_ori = oq_list[qi]
            for cand_qid in cache['orig_to_cot_ids'].get(sim_ori, []):
                cand_cid = cache['query_to_content'][cand_qid]
                if cand_qid not in hard and cand_qid != qid and cand_cid != pos_cid:
                    hard.append(cand_qid)
                    cnt += 1
                    break
        if cnt >= 2:
            break
    # Step 4: Query→CoT 检索补齐
    rem = topk - len(hard)
    if rem > 0:
        qe = cache['query_id_to_embedding'][qid].reshape(1, -1)
        sims2, idxs2 = cache['cot_index'].search(qe, 100)
        for i in idxs2[0]:
            cand_qid = cache['cot_ids'][i]
            cand_cid = cache['query_to_content'][cand_qid]
            if cand_qid != qid and cand_cid != pos_cid and cand_qid not in hard:
                hard.append(cand_qid)
                if len(hard) >= topk:
                    break

    return hard[:topk]

  

# 改进后的主处理函数
def process_all_queries(query_df):
    cot_cache = build_indices_and_caches(query_df)

    results_hard = []
    
    
    for idx, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Processing queries"):
        print(f"\n处理查询 {idx+1}/{len(query_df)}: {row['query_id']}")
        try:
            # 生成hard负样本
            hard_negatives = generate_cot_hard_negatives(row, cot_cache)
            results_hard.append({
                'query_id': row['query_id'],
                'content_id': row['content_id'],  # 正样本
                'hard_cots_qid': ",".join([ str(i) for i in  hard_negatives] ),  # 难负样本
                'hard_negative_count': len(hard_negatives)
            })
            print(f"为查询 {row['query_id']} 生成了 {len(hard_negatives)} 个hard负样本")
    

        except Exception as e:
            print(f"处理查询 {row['query_id']} 时出错: {str(e)}")
            
    
    # 转换为DataFrame
    results_df_hard = pd.DataFrame(results_hard)
    return results_df_hard


# 使用示例
def main():
    # 加载数据
    # 假设这些数据已经存在
    query_df = pd.read_csv('../input/synthetic.csv')
    
    # 处理一小部分查询进行测试
    # print("处理小样本测试...")
    # sample_results = process_all_queries(query_df, content_pool_df, sample_count=5)
    
    # # 保存结果
    # sample_results.to_csv('hard_negatives_sample.csv', index=False)
    
    # 处理所有查询
    print("\n处理所有查询...")
    full_results_hard = process_all_queries(query_df)

    full_results_hard.to_csv('negatives_cot.csv', index=False)

    print("难负样本生成完成！")


if __name__ == "__main__":
    main()