"""
PacketSentry 特征提取核心模块

将数据包按时间窗口切分，提取多维统计特征向量，
作为机器学习模型的输入。
"""

from dataclasses import dataclass
from typing import Optional

from packetsentry.collector.parser import ParsedPacket
from packetsentry.features.statistics import StatisticsCalculator
from packetsentry.features.encoder import FeatureEncoder


@dataclass
class FeatureWindow:
    """时间窗口特征

    封装一个时间窗口内的所有特征和元数据。
    """
    window_start: float                   # 窗口开始时间
    window_end: float                     # 窗口结束时间
    packet_count: int                     # 窗口内数据包数
    features: dict[str, float]            # 特征字典
    is_anomaly: Optional[bool] = None     # 是否异常（检测结果填充）


class FeatureExtractor:
    """特征提取器

    将数据包列表按时间窗口切分，对每个窗口计算
    多维统计特征，生成特征向量序列。
    """

    def __init__(
        self,
        window_size: float = 5.0,
        window_step: float = 2.5,
    ) -> None:
        """初始化特征提取器

        Args:
            window_size: 时间窗口大小（秒）
            window_step: 窗口滑动步长（秒）
        """
        self.window_size = window_size
        self.window_step = window_step
        self.calculator = StatisticsCalculator()
        self.encoder = FeatureEncoder()

    def extract(self, packets: list[ParsedPacket]) -> list[FeatureWindow]:
        """从数据包列表提取时间窗口特征

        将数据包按时间排序，按窗口大小切分，
        对每个窗口计算全部统计特征。

        Args:
            packets: 数据包列表

        Returns:
            FeatureWindow 列表
        """
        if not packets:
            return []

        # 按时间排序
        sorted_packets = sorted(packets, key=lambda p: p.timestamp)
        start_time = sorted_packets[0].timestamp
        end_time = sorted_packets[-1].timestamp

        windows: list[FeatureWindow] = []
        window_start = start_time

        while window_start < end_time:
            window_end = window_start + self.window_size

            # 收集窗口内的数据包
            window_packets = [
                p for p in sorted_packets
                if window_start <= p.timestamp < window_end
            ]

            if window_packets:
                features = self._compute_window_features(window_packets)
                windows.append(FeatureWindow(
                    window_start=window_start,
                    window_end=window_end,
                    packet_count=len(window_packets),
                    features=features,
                ))

            window_start += self.window_step

        return windows

    def _compute_window_features(self, packets: list[ParsedPacket]) -> dict[str, float]:
        """计算单个时间窗口的全部特征

        整合各统计计算器的结果为一个特征字典。

        Args:
            packets: 窗口内的数据包列表

        Returns:
            完整特征字典
        """
        features: dict[str, float] = {}

        # 1. 流量量与方向统计
        features.update(self.calculator.compute_traffic_direction_stats(packets))

        # 2. 包到达间隔统计
        features.update(self.calculator.compute_inter_arrival_stats(packets))

        # 3. 包大小统计
        features.update(self.calculator.compute_packet_size_stats(packets))

        # 4. 协议分布
        features.update(self.calculator.compute_protocol_distribution(packets))

        # 5. TCP 标志位分布
        features.update(self.calculator.compute_tcp_flags_distribution(packets))

        # 6. IP 多样性
        features.update(self.calculator.compute_ip_diversity(packets))

        return features

    def extract_feature_matrix(
        self, packets: list[ParsedPacket]
    ) -> tuple[list[FeatureWindow], list[dict[str, float]]]:
        """提取特征矩阵（用于模型训练/预测）

        Args:
            packets: 数据包列表

        Returns:
            (FeatureWindow列表, 特征字典列表)
        """
        windows = self.extract(packets)
        feature_dicts = [w.features for w in windows]
        return windows, feature_dicts
