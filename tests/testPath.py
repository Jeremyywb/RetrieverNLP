from ..config.configs import  RetrieverModelConfig
from ..utils.utilities import ProjPathsFiles
def testRetrieverModelConfig():
    args_dict = {'model_name': 'nice',
    'model_path': "resource/bge_model_pt",
    'output_dir':"resource/bge_model_ft",
    'use_inbatch_neg': True,
    'temperature': 1 }
    print("test,.....")
    trLoaderConfig = RetrieverModelConfig.from_dict(args_dict)
    print(trLoaderConfig)

    print(ProjPathsFiles(args_dict['model_path']).abs_path_or_file)
    return trLoaderConfig

if __name__ == '__main__':
    print(1233444)
    testRetrieverModelConfig()