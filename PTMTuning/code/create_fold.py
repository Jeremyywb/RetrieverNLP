import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def generate_stratified_folds(query_df, n_folds=5):
    """
    生成分层交叉验证的fold，保证：
    1. 相同original_query_id的样本在同一fold
    2. 每个fold的样本量尽可能均衡
    3. 考虑不同original_query_id的样本数量分布
    
    参数：
    query_df : DataFrame 必须包含original_query_id列
    n_folds : int fold数量，默认为5
    
    返回：
    包含fold_id的DataFrame
    """
    # 创建分组统计信息
    group_info = query_df.groupby('original_query_id').agg(
        group_size=('query_id', 'count'),
        query_ids=('query_id', list)
    ).reset_index()
    
    # 根据组大小分层
    group_size_bins = pd.cut(group_info['group_size'], 
                           bins=[0,1,2,3,np.inf],
                           labels=['1','2','3','4+'])
    group_info['strata'] = group_size_bins.astype(str)
    
    # 初始化分层KFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 为每个group分配fold
    group_info['fold_id'] = -1
    for fold, (_, test_idx) in enumerate(skf.split(X=group_info, y=group_info['strata'])):
        group_info.iloc[test_idx, group_info.columns.get_loc('fold_id')] = fold
    
    # 展开到原始query_id
    fold_map = []
    for _, row in group_info.iterrows():
        for qid in row['query_ids']:
            fold_map.append({'query_id': qid, 'fold_id': row['fold_id']})
    
    fold_df = pd.DataFrame(fold_map)
    
    # 合并到原始数据
    query_df_fold = query_df.merge(fold_df, on='query_id', how='left')
    
    return query_df_fold

# 使用示例
import pandas as pd 

DATAPATH = '../input/synthetic.csv'
df_query = pd.read_csv(DATAPATH)  


DATAPATH = '../input/fold_df.csv'
df_fold = pd.read_csv(DATAPATH)  


df_query['original_query_id'] = df_query['query_id'].str.split('_').str[0]

# 生成fold
fold_df = generate_stratified_folds(df_query)

# 验证分布
print(fold_df.groupby('fold_id').agg(
    num_queries=('query_id', 'count'),
    num_groups=('original_query_id', 'nunique'),
    avg_group_size=('original_query_id', lambda x: fold_df.groupby('original_query_id')['query_id'].count().mean())
))

save_cols = ['query_id','original_query_id','fold_id']
fold_df[save_cols].to_csv("../input/fold_df.csv",index=False)