


# # pip install faiss-gpu==1.7.2
# class CFG:
#     input_path = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/"
#     train_path = f"{input_path}train.csv"
#     test_path  = f"{input_path}test.csv"
#     misc_path  = f"{input_path}misconception_mapping.csv"
#     samp_path  = f"{input_path}sample_submission.csv"
# #     max_cutoff = 50
#     max_cutoff = 100
#     is_add_inner_pos_ids = False
    
#     is_train   = False
#     if is_train:
#         embd_name  = "BAAI/bge-large-en-v1.5"#online
#         rerank_na  = 'BAAI/bge-reranker-large'#online
# #         embd_name  = "Alibaba-NLP/gte-large-en-v1.5"

#     else:
#         embd_name  = "/kaggle/input/bge-large-en-v1-5/bge-large-en-v1.5"#offline
#         emb_finetune = "/kaggle/input/bge-retriever-ft-v1/resource/bge-emb-v1-ft"
#         rerank_na  = "/kaggle/input/bge-reranker-large"#offline
        
        
#     #========================================
#     # set up for hard negative sample mining
#     #========================================
#     model_name_or_path = embd_name
#     output_path = '/kaggle/working/output/'

    
def format_text(all_text,option_text):
    return f'''Question: {all_text}\n\n Option: {option_text}.\n\n What misconception does this option reveal?'''

def prepare_for_hard_mining(cfg, config ):
    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    from scipy.spatial.distance import cdist
    import json
    import os
    import numpy as np
    from tqdm import tqdm
    from .retriever_pred import RetrieverInfference
    if not os.path.exists(cfg.output_path):
        os.mkdir( cfg.output_path )

    RetrieverInffer = RetrieverInfference(config)

    train                 = pd.read_csv(cfg.train_path)
    misconception_mapping = pd.read_csv(cfg.misc_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.embd_name)

    def make_all_question_text(df: pd.DataFrame) -> pd.DataFrame:
        df["all_question_text"] = df["ConstructName"] +". " +df["QuestionText"]
        return df
    train = make_all_question_text(train)

    def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        df = pd.melt(
        df[
            [
                "QuestionId",
                "all_question_text",
                "CorrectAnswer",
                "AnswerAText",
                "AnswerBText",
                "AnswerCText",
                "AnswerDText"
            ]
        ],
            id_vars    = ["QuestionId", "all_question_text", "CorrectAnswer"],
            var_name   = 'Answer',
            value_name = 'value'
        )

        return df

    train_long = wide_to_long(train)
    train_long['answer_text'] = train_long['Answer'].map(lambda x: x.replace("Answer","").replace("Text",""))
    questionid_to_correct_text = train_long[
            train_long['answer_text'] == train_long['CorrectAnswer'] 
            ].set_index('QuestionId')['value'].to_dict()
    train_long['correct_text'] = train_long['QuestionId'].map(questionid_to_correct_text)

    def make_all_text(df: pd.DataFrame) -> pd.DataFrame:
        df["all_text"] =("Question: " + df["all_question_text"] +
                         ".\n\n Correct Option: " + df["correct_text"] +
                          ".\n\n Wrong Option: " + df["value"] + ".\n\n What misconception does wrong option reveal?")
        return df

    train_long = make_all_text(train_long)

    def drop_correct_text(df):
        df = df[ df.answer_text!=df.CorrectAnswer ].reset_index(drop=True)
        return df
    train_long = drop_correct_text(train_long)
    labels = misconception_mapping['MisconceptionName'].values


    def get_misconceptionName_dict(df, MisconceptionName):
        misconception_name_dict = {}
        for index, row in df.iterrows():
            QuestionId = row.QuestionId
            for answer in list("ABCD"):
                correct_answer = row.CorrectAnswer
                if answer != correct_answer:
                    key =  f"{QuestionId}_{answer}"
                    val = row.get(f"Misconception{answer}Id")
                    if pd.isna(val):
                        continue
                    else:
                        misconception_name_dict[key] = MisconceptionName[int(val)]
        return misconception_name_dict

    def get_misconceptionID_dict(df, MisconceptionName):
        misconception_name_dict = {}
        for index, row in df.iterrows():
            QuestionId = row.QuestionId
            for answer in list("ABCD"):
                correct_answer = row.CorrectAnswer
                if answer != correct_answer:
                    key =  f"{QuestionId}_{answer}"
                    val = row.get(f"Misconception{answer}Id")
                    if pd.isna(val):
                        continue
                    else:
                        misconception_name_dict[key] = int(val)
        return misconception_name_dict

    id_to_name = get_misconceptionName_dict(train, labels)
    id_to_msid = get_misconceptionID_dict(train, labels)
    train_long['QuestionId_Answer'] = train_long.QuestionId.map(str) + "_" +train_long['Answer'].map(
            lambda x:x.replace("Answer","").replace("Text","")
    )
    train_long['pos_text'] = train_long['QuestionId_Answer'].map(id_to_name)
    train_long = train_long[train_long['pos_text'].isnull()==False]
    train_long['pos_id'  ] = train_long['QuestionId_Answer'].map(id_to_msid)
    questionid_to_list_of_pos_ids = train_long.groupby('QuestionId')['pos_id'].apply(list).to_dict()


    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg.embd_name, output_hidden_states=True)
    model     = AutoModel.from_pretrained(cfg.embd_name, config=config)

    from transformers import AutoTokenizer, AutoModel
    import torch


    device = "cuda:0"


    model.eval()
    model.to(device)
    print("finish")



    # ========================
    # Misconception embedding
    # ========================

    from tqdm import tqdm
    MisconceptionName = list(misconception_mapping['MisconceptionName'].values)
    per_gpu_batch_size = 8


    def prepare_inputs(text, tokenizer, device):
        tokenizer_outputs = tokenizer.batch_encode_plus(
            text,
            padding        = True,
            return_tensors = 'pt',
            max_length     = 1024,
            truncation     = True
        )
        result = {
            'input_ids': tokenizer_outputs.input_ids.to(device),
            'attention_mask': tokenizer_outputs.attention_mask.to(device),
        }
        return result


    all_ctx_vector = []
    for mini_batch in tqdm(range(0, len(MisconceptionName[:]), per_gpu_batch_size)):
        mini_context          = MisconceptionName[mini_batch:mini_batch+ per_gpu_batch_size]
        encoded_input         = prepare_inputs(mini_context,tokenizer,device)

        sentence_embeddings = RetrieverInffer.encode_passages(encoded_input)
        all_ctx_vector.append(sentence_embeddings.detach())
        del sentence_embeddings,encoded_input
        torch.cuda.empty_cache()

        # sentence_embeddings   = model(**encoded_input)[0][:, 0]
        # sentence_embeddings   = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # all_ctx_vector.append(sentence_embeddings.detach().cpu().numpy())
        # del sentence_embeddings

    # all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
    all_ctx_vector = torch.cat(all_ctx_vector, axis=0)



    # ========================
    # train query embedding
    # ========================
    train_texts = list(train_long.all_text.values)
    train_text_vector = []
    per_gpu_batch_size = 8

    for mini_batch in tqdm(
            range(0, len(train_texts[:]), per_gpu_batch_size)):
        mini_context = train_texts[mini_batch:mini_batch
                                            + per_gpu_batch_size]
        encoded_input = prepare_inputs(mini_context,tokenizer,device)
        sentence_embeddings = RetrieverInffer.encode(encoded_input)
        train_text_vector.append(sentence_embeddings.detach())
        del sentence_embeddings,encoded_input


        # sentence_embeddings = model(
        #     **encoded_input)[0][:, 0]
        # sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # train_text_vector.append(sentence_embeddings.detach().cpu().numpy())
        # del sentence_embeddings

    # train_text_vector = np.concatenate(train_text_vector, axis=0)
    train_text_vector = torch.cat(train_text_vector, axis=0)
    print(train_text_vector.shape)

    # ========================
    # Cosine similarities
    # ========================

    # cosine_similarities = cosine_similarity(train_text_vector, all_ctx_vector)
    # train_sorted_indices = np.argsort(-cosine_similarities, axis=1)


    # ========================
    # model inner similarities
    # ========================
    similarity = None
    for i in tqdm(
            range(0,train_text_vector.shape[0], 32 ) ):
        
        sentence_embeddings = RetrieverInffer.compute_similarity(train_text_vector[i:i+32], all_ctx_vector )
        if similarity is None:
            similarity = sentence_embeddings
        else:
            similarity = torch.cat([similarity, sentence_embeddings],dim=0)
        del sentence_embeddings
        torch.cuda.empty_cache()

    train_sorted_indices = np.argsort(-similarity.detach().cpu().numpy(), axis=1)
    del similarity



    # ========================
    # Hard negative sample mining
    # ========================
    '''
    write json lines
    {   
        "query": "query",
        "pos": ["pos"],
        "inner_neg_end": int=3?,
        "docs": ["doc1","doc2","doc3"..],
        "pos_mask": [1,0,0,...]
    }
    '''
    

    train_sorted_indices_to_list = train_sorted_indices.tolist()
    questionid_list = train_long.QuestionId.values.tolist()
    pos_ids_list = train_long.pos_id.values.tolist()


    all_data = {}
    for i in range(len(train_sorted_indices_to_list)):
        questionid = questionid_list[i]
        pos_id = pos_ids_list[i]
        other_neg_ids = questionid_to_list_of_pos_ids[questionid]
        other_neg_ids = [x for x in other_neg_ids if x != pos_id]
        sorted_indices = train_sorted_indices_to_list[i]
        if len(other_neg_ids) >0 and cfg.is_add_inner_pos_ids:
            sorted_indices = other_neg_ids + sorted_indices
        #pos_mask  like == [0,0,0,1,0,0,0]
        pos_mask = [0] * len(sorted_indices)
        pos_mask[sorted_indices.index(pos_id)] = 1
        sorted_indices = sorted_indices[:cfg.max_cutoff]
        pos_mask = pos_mask[:cfg.max_cutoff] #检查是不是64
        query = train_texts[i]
        sorted_indices_text = [MisconceptionName[x] for x in sorted_indices]
        data = {
            "query": query,
            "pos": [MisconceptionName[pos_id]],
            "docs": sorted_indices_text,
            "pos_mask": pos_mask,
            'passag_id': pos_id,
            'inner_neg_end': len(other_neg_ids)
        }
        _hash = hash(query)
        all_data[_hash] = data

    def load_json_lines(json_file):
        data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    train_data = load_json_lines(cfg.last_train )
    for i in range(len(train_data)):
        dat = train_data[i]
        qry = dat['query']
        _hash_docs = [ hash(d) for d in dat['docs'] ]
        _hash = hash(qry)
        _add  = all_data[_hash]
        _add_pos = _add['pos'][0]
        _add_doc = _add['docs']
        _add_doc = [c for c in _add_doc if hash(c) != hash(_add_pos) and hash(c) not in _hash_docs]

        if len(_add_doc)<len(_hash_docs) :
            _len_save = len(_hash_docs)//2
            _len_add  = len(_hash_docs) - _len_save
            if _len_add<len(_add_doc):
                dat['docs'] = dat['docs'][:_len_save] + _add_doc[:_len_add]
                train_data[i] = dat
        else:
            _add['docs'] = _add_doc
            _add['pos_mask'] = [0]*len(_add_doc)
            train_data.append(_add)
    
    with open(f"{cfg.output_path}{cfg.output_train}.json", "w") as f:
        for data in train_data:
            f.write(json.dumps(data) + "\n")

