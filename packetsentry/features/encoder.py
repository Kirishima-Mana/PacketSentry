"""
PacketSentry 特征编码与归一化模块

将提取的原始特征进行编码、归一化和对齐，
生成机器学习模型可直接使用的特征向量。
"""

import numpy as np
import pandas as pd
from typing import Optional


# 特征列名列表（与特征提取器输出对齐）
FEATURE_COLUMNS: list[str] = [
    # 包速率与流量量
    "total_packets", "total_bytes", "avg_pkt_size",
    "payload_bytes", "ratio_with_payload", "ratio_broadcast",
    # 包到达间隔统计
    "iat_mean", "iat_std", "iat_max", "iat_min",
    # 包大小统计
    "pkt_size_mean", "pkt_size_std", "pkt_size_max", "pkt_size_min",
    # 协议分布
    "ratio_tcp", "ratio_udp", "ratio_icmp", "ratio_other",
    # TCP 标志位分布
    "ratio_syn", "ratio_ack", "ratio_fin", "ratio_rst", "ratio_psh",
    # IP 多样性
    "unique_src_ip", "unique_dst_ip", "unique_dst_port",
    "src_ip_entropy", "dst_port_entropy", "packets_per_src_ip",
]


class FeatureEncoder:
    """特征编码器

    将原始特征字典转换为标准化的特征向量，
    处理缺失值和异常值。
    """

    def __init__(self, feature_columns: Optional[list[str]] = None) -> None:
        """初始化编码器

        Args:
            feature_columns: 特征列名列表，None 则使用默认
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def encode(self, feature_dict: dict[str, float]) -> np.ndarray:
        """将特征字典编码为特征向量

        按照预定义的列顺序提取特征值，缺失值填充为 0。

        Args:
            feature_dict: 原始特征字典

        Returns:
            特征向量（1D numpy 数组）
        """
        vector = np.zeros(len(self.feature_columns), dtype=np.float64)
        for i, col in enumerate(self.feature_columns):
            val = feature_dict.get(col, 0.0)
            # 处理 NaN 和 Inf
            if not np.isfinite(val):
                val = 0.0
            vector[i] = val
        return vector

    def encode_batch(self, feature_dicts: list[dict[str, float]]) -> np.ndarray:
        """批量编码特征字典

        Args:
            feature_dicts: 特征字典列表

        Returns:
            特征矩阵（2D numpy 数组，shape: [n_samples, n_features]）
        """
        if not feature_dicts:
            return np.zeros((0, len(self.feature_columns)), dtype=np.float64)

        matrix = np.zeros(
            (len(feature_dicts), len(self.feature_columns)), dtype=np.float64
        )
        for i, fd in enumerate(feature_dicts):
            matrix[i] = self.encode(fd)
        return matrix

    def fit_normalize(self, feature_matrix: np.ndarray) -> None:
        """拟合归一化参数（均值和标准差）

        使用训练数据计算 Z-Score 归一化所需的统计量。

        Args:
            feature_matrix: 训练特征矩阵
        """
        self._means = np.nanmean(feature_matrix, axis=0)
        self._stds = np.nanstd(feature_matrix, axis=0)
        # 防止除以零
        self._stds[self._stds == 0] = 1.0

    def normalize(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Z-Score 归一化

        使用 fit_normalize 计算的均值和标准差进行归一化。

        Args:
            feature_matrix: 待归一化的特征矩阵

        Returns:
            归一化后的特征矩阵

        Raises:
            ValueError: 未先调用 fit_normalize
        """
        if self._means is None or self._stds is None:
            raise ValueError("请先调用 fit_normalize() 拟合归一化参数")
        return (feature_matrix - self._means) / self._stds

    def to_dataframe(self, feature_dicts: list[dict[str, float]]) -> pd.DataFrame:
        """将特征字典列表转换为 DataFrame

        Args:
            feature_dicts: 特征字典列表

        Returns:
            pandas DataFrame
        """
        return pd.DataFrame(feature_dicts, columns=self.feature_columns)
