# This is the test_training.py file in tests folder
from ..training.trainer import Trainer
import pprint


def test_trainer(training_config_yaml_or_dict):
    trainer = Trainer(training_config_yaml_or_dict)
    args = trainer.args
    return args

def print_args(args):
    for key in args:
        print(f"=============={key}==============")
        # 使用 pprint 来格式化输出
        formatted_value = pprint.pformat(args[key], indent=4, width=1)
        # 删除开头和结尾的花括号
        formatted_value = formatted_value.strip('{}')
        # 在每行前面添加四个空格
        formatted_value = '\n'.join('    ' + line for line in formatted_value.split('\n'))
        # 重新添加类名和括号
        class_name = args[key].__class__.__name__
        print(f"{class_name}(\n{formatted_value}\n)")


# if __name__ == "__main__":
#     test_trainer("resource/args-yaml/bge-emebeding-training-config-V1.yaml")

