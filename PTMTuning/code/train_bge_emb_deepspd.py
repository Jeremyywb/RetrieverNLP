from copy import deepcopy
from dataclasses import dataclass
import json
import time
from transformers.utils import logging as hf_logging
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import hydra
import os
import pandas as pd
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# local import 
from bge_embedding.ptm_model import get_base_model,BgeBiEncoderModel,add_paths_to_config
from bge_embedding.ptm_optimizer import get_optimizer
from bge_embedding.ptm_dataset import get_tokenizer,QueryDataset,ContentDataset,CotDataset,RetrieverDataset
from bge_embedding.ptm_dataloader import TripletCollator,show_batch,show_batch_fs,TextCollator

from utils.train_utils import (as_minutes, 
     load_ext_cot, 
     get_custom_cosine_schedule_with_warmup, 
    get_cosine_schedule_with_warmup_and_minlr,
     get_lr, 
     setup_training_run, 
     train_valid_split
     )

from utils.retriever_utils import semantic_search
from utils.metric_utils import compute_retrieval_metrics, mapk


@dataclass
class IDTracker:
    """Track different IDs during training and evaluation"""

    query_train_ids: list
    query_valid_ids: list
    content_train_ids: list
    content_comp_ids: list

# -------- Evaluation -----------------------------------------------------------------------------#


def run_evaluation(cfg, accelerator, model, query_dl, content_dl, label_df, id_tracker):
    cutoffs = [1, 2, 4, 8, 16, 25, 32, 64]
    label_df = deepcopy(label_df)
    query2content = label_df.groupby("query_id")["content_id"].apply(list).to_dict()
    # query_id:str-->[content_id]:int

    model.eval()

    query_embeddings = []
    progress_bar = tqdm(range(len(query_dl)))
    for batch in query_dl:
        with torch.no_grad():
            batch_query_embeddings = accelerator.unwrap_model(model).encode(batch,mode='q')

        batch_query_embeddings = accelerator.gather_for_metrics(batch_query_embeddings)
        query_embeddings.append(batch_query_embeddings)
        progress_bar.update(1)
    progress_bar.close()

    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_ids = id_tracker.query_valid_ids
    accelerator.print(f"shape of query embeddings: {query_embeddings.shape}")
    assert query_embeddings.shape[0] == len(query_ids)

    # get content embeddings ---
    content_embeddings = []
    progress_bar = tqdm(range(len(content_dl)))

    for batch in content_dl:
        with torch.no_grad():
            batch_content_embeddings = accelerator.unwrap_model(model).encode(batch,mode='r')
        batch_content_embeddings = accelerator.gather_for_metrics(batch_content_embeddings)
        content_embeddings.append(batch_content_embeddings)
        progress_bar.update(1)
    progress_bar.close()

    content_embeddings = torch.cat(content_embeddings, dim=0)
    content_ids = id_tracker.content_comp_ids # int
    accelerator.print(f"shape of content embeddings: {content_embeddings.shape}")
    assert content_embeddings.shape[0] == len(content_ids)

    # ------ evaluation ----------------------------------------------------------------#
    results = semantic_search(query_embeddings, content_embeddings, top_k=cfg.model.n_neighbour)

    true_content_ids = []
    pred_content_ids = []
    pred_scores = []

    for idx, re_i in enumerate(results):  # loop over query
        query_id = query_ids[idx]
        hit_i = [node["corpus_id"] for node in re_i]
        top_scores_i = [node["score"] for node in re_i]
        top_content_ids_i = [content_ids[pos] for pos in hit_i]
        pred_content_ids.append(top_content_ids_i)
        pred_scores.append(top_scores_i)
        true_content_ids.append(query2content[query_id])

    result_df = pd.DataFrame()
    result_df["query_id"] = query_ids
    result_df["true_ids"] = true_content_ids
    result_df["pred_ids"] = pred_content_ids
    result_df["pred_scores"] = pred_scores

    # compute metric ----
    eval_dict = dict()

    for cutoff in cutoffs:
        cdf = result_df.copy()
        cdf["pred_ids"] = cdf["pred_ids"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)

        eval_dict[f"precision@{cutoff}"] = m["precision_score"]
        eval_dict[f"recall@{cutoff}"] = m["recall_score"]

    # get mapk ---
    eval_dict["lb"] = mapk(result_df["true_ids"].values, result_df["pred_ids"].values, k=25)
    accelerator.print(f">>> LB: {eval_dict['lb']}")

    # seen vs unseen
    content_train_ids = id_tracker.content_train_ids
    result_df["seen"] = result_df["true_ids"].apply(lambda x: True if x[0] in content_train_ids else False)

    seen_df = result_df[result_df["seen"]].reset_index(drop=True)
    unseen_df = result_df[~result_df["seen"]].reset_index(drop=True)

    eval_dict["seen_lb"] = mapk(seen_df["true_ids"].values, seen_df["pred_ids"].values, k=25)
    eval_dict["unseen_lb"] = mapk(unseen_df["true_ids"].values, unseen_df["pred_ids"].values, k=25)

    # get oof df
    oof_df = result_df.copy()
    oof_df = oof_df.drop(columns=["true_ids"])
    oof_df = oof_df.rename(columns={"query_id": "QuestionId_Answer"})
    oof_df = oof_df.rename(columns={"pred_ids": "MisconceptionId"})
    oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(lambda x: list(map(str, x)))
    oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(lambda x: " ".join(x))

    to_return = {"lb": eval_dict["lb"], "scores": eval_dict, "result_df": result_df, "oof_df": oof_df}

    # logs -----
    scores_dict = eval_dict
    accelerator.print("--------------------------------")
    accelerator.print(f">>> LB: {scores_dict['lb']}")
    accelerator.print(f">>> Seen LB: {scores_dict['seen_lb']}")
    accelerator.print(f">>> Unseen LB: {scores_dict['unseen_lb']}")
    accelerator.print("--------------------------------")

    for pt in cutoffs:
        accelerator.print(f">>> Current Recall@{pt} = {round(scores_dict[f'recall@{pt}'], 4)}")

    return to_return


# -------- Main Function --------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../conf/bge_embedding", config_name="conf_qcot")
def run_training(cfg):
    accelerator = setup_training_run(cfg)
    cfg = add_paths_to_config(cfg)

    def print_line(print_fn=accelerator.print):
        prefix, unit, suffix = "#", "~~", "#"
        print_fn(prefix + unit * 50 + suffix)

    cfg.local_rank = accelerator.process_index


    # ------- load data --------------------------------------------------------------------------#
    with accelerator.main_process_first():
        df = pd.read_csv(cfg.dataset.query_dataset)
        content_df = pd.read_csv(cfg.dataset.content_dataset)
        content_df = content_df.rename(columns = {'MisconceptionId':'content_id'})
        cot_ext_df = load_ext_cot(cfg.dataset.ext_cot_datasdet)
        cot_neg_df = pd.read_csv(cfg.dataset.negative_cot_dataset)
        # content_df_comp = pd.read_csv(os.path.join(data_dir_comp, "misconception_mapping.csv")).rename(columns={"MisconceptionId": "content_id"})
        train_df, valid_df = train_valid_split(cfg, df)
        negative_df = None
        if cfg.dataset.negative_dataset != '...':
            negative_df = pd.read_csv(cfg.dataset.negative_dataset)

    accelerator.wait_for_everyone()

    # process data --------------------------------------------------------------------------------#
    if negative_df is not None:
        query_to_content_ids =  negative_df.groupby(
            "query_id")[f"{cfg.task.name}_negatives"].apply(
                lambda x: x.str.split(',').explode().tolist()
            ).to_dict()
    else:
        query_to_content_ids = None
    query_to_cot_query_ids =  cot_neg_df.groupby(
            "query_id")["hard_cots_qid"].apply(
                lambda x: x.str.split(',').explode().tolist()
            ).to_dict()
    print_line()
    accelerator.print(f"# of queries (train): {train_df.shape[0]}")
    accelerator.print(f"# of queries (valid): {valid_df.shape[0]}")
    accelerator.print(f"# shape of content data: {content_df.shape}")
    accelerator.print(f"# shape of cot data(ext cot): {cot_ext_df.shape}")
    print_line()

    # ------- Datasets ----------------------------------------------------------------------------#
    tokenizer = get_tokenizer(cfg)
    train_query_dataset = QueryDataset(
      train_df, 
      tokenizer=tokenizer, 
      max_length=cfg.model.max_length
    )

    valid_query_dataset = QueryDataset(
      valid_df, 
      tokenizer=tokenizer, 
      max_length=cfg.model.max_length
    )

    cot_dataset = CotDataset(
        df, 
        tokenizer=tokenizer, 
        max_length=cfg.model.max_length
    )
    cot_ext_dataset = CotDataset(
        cot_ext_df, 
        tokenizer=tokenizer, 
        max_length=cfg.model.max_length
    )

    content_dataset = ContentDataset(
        content_df, 
        tokenizer=tokenizer, 
        max_length=cfg.model.max_length
    )
    retriever_dataset = RetrieverDataset(
            cfg,
            query_dataset=train_query_dataset,
            content_dataset=content_dataset,
            cot_dataset=cot_dataset,
            external_cot_dataset=cot_ext_dataset,
            cot_negatives = query_to_cot_query_ids,
            negatives=query_to_content_ids
            )
    
    
    # manage ids ---
    id_tracker = IDTracker(
        query_train_ids=train_df["query_id"],
        query_valid_ids=valid_df["query_id"],
        content_train_ids=train_df["content_id"],
        content_comp_ids=content_df["content_id"],
    )




    
    # ------- data collators ----------------------------------------------------------------------#
    tri_collator = TripletCollator()
    text_collator = TextCollator()
    

    # ------- data loaders ------------------------------------------------------------------------#

    query_valid_dl = DataLoader(
        valid_query_dataset,
        batch_size=cfg.train_params.query_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    content_comp_dl = DataLoader(
        content_dataset,
        batch_size=cfg.train_params.content_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    retrieval_dl = DataLoader(
        retriever_dataset,
        batch_size=cfg.train_params.retriever_bs,
        shuffle=True,
        collate_fn=tri_collator,
        num_workers=1,
        drop_last=True,
    )

    # --- show batch ------------------------------------------------------------------------------#
    print_line()

    accelerator.print("showing a batch...")
    for b in retrieval_dl:
        break
    show_batch(b, tokenizer, print_fn=accelerator.print)

    print_line()
    accelerator.print("showing first valid batch...")
    for b in query_valid_dl:
        break
    show_batch_fs(b, tokenizer, print_fn=accelerator.print)
    print_line()

    # ------- Config ------------------------------------------------------------------------------#
    print_line()
    accelerator.print("Config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    accelerator.print(json.dumps(cfg_dict, indent=4))
    print_line()

    # ------- Model -------------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    base_model, head_model  = get_base_model(cfg)
    
    print('#---IN BASE MODEL-----------------------------------------------')
    base_model.train()
    for name, param in base_model.named_parameters():
        if "lora" in name.lower():
            if param.requires_grad:
                print("✅ Found LoRA param:", name)
                print(name, "requires_grad:", param.requires_grad)
                break
    base_model.eval()
    print('#---IN BASE MODEL-----------------------------------------------')
    
    
    model = BgeBiEncoderModel(cfg, base_model, head_model, accelerator)

    print('#---IN BgeBiEncoderModel MODEL-----------------------------------------------')

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            if param.requires_grad:
                print("✅ Found LoRA param:", name)
                print(name, "requires_grad:", param.requires_grad)
                break

    print('#---IN BgeBiEncoderModel MODEL-----------------------------------------------')
    a = 1



    if cfg.model.gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.model.compile:
        accelerator.print("compiling model")
        model = torch.compile(model)
    accelerator.print("Model loaded")
    print_line()

    # ------- Optimizer ---------------------------------------------------------------------------#
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    print('#---IN BASE MODEL-----------------------------------------------')

    for name, param in base_model.named_parameters():
        if "lora" in name.lower():
            if param.requires_grad:
                print("✅ Found LoRA param:", name)
                print(name, "requires_grad:", param.requires_grad)
                break

    print('#---IN BASE MODEL-----------------------------------------------')


    
    # ------- Accelerator -------------------------------------------------------------------------#
    model, optimizer, query_valid_dl, retrieval_dl, content_comp_dl = accelerator.prepare(
        model, optimizer, query_valid_dl, retrieval_dl, content_comp_dl
    )
    # ------- Scheduler ---------------------------------------------------------------------------#
    print_line()

    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(retrieval_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")
    
    # scheduler = get_custom_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scheduler = get_cosine_schedule_with_warmup_and_minlr(
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        min_lr = cfg.optimizer.lr*0.01
    )
    
    
    # ------- training setup ----------------------------------------------------------------------#
    best_lb = -1.0
    patience_tracker = 0
    current_iteration = 0
    progress = 0.0

    start_time = time.time()
    progress_bar = None

    # initial evaluation --------------------------------------------------------------------------#
    if cfg.train_params.eval_at_start:
        model.eval()
        eval_response = run_evaluation(cfg, accelerator, model, query_valid_dl, content_comp_dl, valid_df, id_tracker)

    for epoch in range(num_epochs):
        if progress_bar:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        print_line()
        accelerator.print(f"Current epoch: {epoch+1}/{num_epochs}")
        print_line()  
                            
        # Training ------
        model.train()
        for step, batch in enumerate(retrieval_dl):
            if not cfg.use_deepspeed_plugin:
                raise ValueError("这个代码必须跑deepspeed开启 cfg.use_deepspeed_plugin 设置开启")
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # if accelerator.sync_gradients:
            progress_bar.set_description(
                f"STEP: {step+1:5}/{num_update_steps_per_epoch:5}. "
                f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                f"LR: {get_lr(optimizer):.4f}. "
                f"Loss: {loss.item():.4f}. "
            )

            progress_bar.update(1)
            current_iteration += 1
            progress = current_iteration / num_training_steps

        # Evaluation -----
        print_line()
        et = as_minutes(time.time() - start_time)
        accelerator.print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")

        model.eval()
        eval_response = run_evaluation(cfg, accelerator, model, query_valid_dl, content_comp_dl, valid_df, id_tracker)

        lb, scores_dict, result_df, oof_df = eval_response["lb"], eval_response["scores"], eval_response["result_df"], eval_response["oof_df"]
        print_line()

        # best scores and saving -----
        is_best = False
        if lb >= best_lb:
            best_lb = lb
            is_best = True
            patience_tracker = 0

            # -----
            best_dict = dict()
            for k, v in scores_dict.items():
                best_dict[f"{k}_at_best"] = v
        else:
            patience_tracker += 1

        if is_best:
            oof_df.to_csv(os.path.join(cfg.paths.save_task_specific_path, "oof_df_best.csv"), index=False)
            result_df.to_csv(os.path.join(cfg.paths.save_task_specific_path, "result_df_best.csv"), index=False)
        else:
            accelerator.print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
            accelerator.print(f">>> current best score: {round(best_lb, 4)}")

        oof_df.to_csv(os.path.join(cfg.paths.save_task_specific_path, "oof_df_last.csv"), index=False)
        result_df.to_csv(os.path.join(cfg.paths.save_task_specific_path, "result_df_last.csv"), index=False)

        # saving -----
        accelerator.wait_for_everyone()

        # save checkpoint ---
        if accelerator.is_main_process:
            if is_best:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save(cfg)
                tokenizer.save_pretrained(cfg.paths.tokenizer_save_path)

        # -- post eval
        model.train()
        torch.cuda.empty_cache()
        print_line()

        # early stopping ----
        if patience_tracker >= cfg.train_params.patience:
            return


if __name__ == "__main__":
    run_training()



