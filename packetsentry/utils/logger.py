"""
PacketSentry 日志模块

提供统一的日志格式化输出，支持 Rich 终端美化。
"""

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logger(
    name: str = "packetsentry",
    level: int = logging.INFO,
    rich_output: bool = True,
) -> logging.Logger:
    """创建并配置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        rich_output: 是否使用 Rich 格式化输出

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    if rich_output:
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)
    return logger


# 全局默认日志器
logger = setup_logger()
