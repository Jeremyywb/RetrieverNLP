import hydra

@hydra.main(version_base=None, config_path="../conf/jina_embedding", config_name="conf_qcot")
def run_training(cfg):
    print(cfg.model.name)
    print(cfg.model.model_name)
    print(cfg.model.backbone_path)
    print(cfg.model.model_type)
    print(cfg.model.trust_remote_code)
    print(cfg.model.backbone_path)  # max_length出
#   name: bge_embedding
#   model_name: BGE-large-en
#   backbone_path: BAAI/bge-large-en-v1.5
#   model_type: bge_embedding
#   trust_remote_code: false
#   max_length: 512
#   sentence_pooling_method: cls
#   gradient_checkpointing: true
if __name__ == "__main__":
    run_training()