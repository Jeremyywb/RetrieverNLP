from ..utils.utilities import ExplicitEnum


    

class ScheType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    # WARMUP_STABLE_DECAY = "warmup_stable_decay"




class ConfigType(ExplicitEnum):
    TRAINSET = 'trainset'
    VALIDSET = 'validset'
    TESTSET = 'testset'
    BGERETRIEVERTRAIN = 'bge_retriever_train'
    BGERETRIEVERVALID = 'bge_retriever_valid'
    BGERETRIEVERTEST = 'bge_retriever_test'
    RERANKERTRAIN = 'reranker_train'
    RERANKERVALID = 'reranker_valid'
    TRAINLOADER = 'train_dataloader'
    VALIDLOADER = 'valid_dataloader'
    TESTLOADER = 'test_dataloader'
    MODEL = 'UNKNOW'
    BGEEMBEDDING = 'bge_embedding'
    BGERERANKER = 'bge_reranker'
    CALLBACKS = 'callbacks'
    SCHEDULER = 'scheulder'
    OPTIMIZER = 'optim'
    TRAINER = 'trainer'




class ModelType(ExplicitEnum):
    NAME = 'ModelType'
    BGEEMBEDDING = 'bge_embedding'
    BGERERANKER = 'bge_reranker'



class DatasetType(ExplicitEnum):
    BGERETRIEVERTRAIN = 'bge_retriever_data_train'
    BGERETRIEVERVALID = 'bge_retriever_data_valid'
    BGERETRIEVERTEST = 'bge_retriever_data_test'
    RERANKERTRAIN = 'bge_reranker_data_train'
    RERANKERVALID = 'bge_reranker_data_valid'


class TrainerConfigType(ExplicitEnum):
    BGERETRIEVERTRAINER = 'bge_retriever_trainer'
    RERANKERTRAINER = 'reranker_trainer'
    
