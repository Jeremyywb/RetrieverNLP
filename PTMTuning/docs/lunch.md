chmod -R 755 /root/cloud/RetrieverNLP/PTMTuning/code/outputs/

accelerate launch --config_file=../conf/acce.yaml train_bge_embedding.py --config-name conf_qcot
accelerate launch --config_file=../conf/acce_deepspeed.yaml train_bge_emb_deepspd.py --config-name conf_qcot_deepspeed


accelerate launch --config_file=../conf/acce_deepspeed.yaml train_bge_emb_deepspd.py --config-name conf_qcot

