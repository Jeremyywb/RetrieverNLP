
报错与hydra有关时大概率是 权限问题，直接linux命令修改权限：
[2025-05-04 16:55:36,801] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-04 16:55:36,820] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.10/logging/config.py", line 565, in configure
    handler = self.configure_handler(handlers[name])
  File "/root/miniconda3/lib/python3.10/logging/config.py", line 746, in configure_handler
    result = factory(**kwargs)
  File "/root/miniconda3/lib/python3.10/logging/__init__.py", line 1169, in __init__
    StreamHandler.__init__(self, self._open())
  File "/root/miniconda3/lib/python3.10/logging/__init__.py", line 1201, in _open
    return open_func(self.baseFilename, self.mode,
FileNotFoundError: [Errno 2] No such file or directory: '/root/cloud/RetrieverNLP/PTMTuning/code/outputs/2025-05-04/16-55-39/train_bge_emb_deepspd.log'


chmod -R 755 /root/cloud/RetrieverNLP/PTMTuning/code/outputs/
并且在训练的配置文件添加：
hydra:
  # 关键点: 使用原始工作目录
  run:
    dir: ${oc.env:HOME}/cloud/RetrieverNLP/PTMTuning/code/outputs/${now:%Y-%m-%d_%H-%M-%S}
  job:
    name: train_sft
    chdir: false  # 不改变工作目录，这是关键配置
  output_subdir: null
  # 日志配置
  job_logging:
    version: 1
    formatters:
      simple:
        format: "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handlers:
      file_handler:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra:run.dir}/train_sft.log
