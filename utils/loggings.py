# This is the logger.py file in utils folder
# This is the loggings.py file in utils folder
import logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    创建并配置一个 logger 实例。

    Args:
        name (str): Logger 的名称，通常使用 __name__。
        level (int): 日志级别，默认为 INFO。

    Returns:
        logging.Logger: 配置好的 logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 检查是否已经有处理器，防止重复添加
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# 可选：设置根 logger 的级别
logging.getLogger().setLevel(logging.WARNING)
