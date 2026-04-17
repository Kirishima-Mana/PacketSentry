"""
PacketSentry 集成决策器

融合 Isolation Forest 和 Autoencoder 的检测结果，
通过多种集成策略产出最终异常判定。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from packetsentry.models.isolation_forest import IsolationForestDetector
from packetsentry.models.autoencoder import AutoencoderDetector
from packetsentry.utils.logger import logger


@dataclass
class EnsembleResult:
    """集成检测结果

    封装双模型的完整检测结果，包括各模型独立判定和集成结论。
    """
    is_anomaly: np.ndarray              # 集成判定：是否异常
    anomaly_scores: np.ndarray          # 集成异常分数（0~1，越高越异常）
    if_scores: np.ndarray               # IF 模型分数
    ae_scores: np.ndarray               # AE 模型分数
    if_predictions: np.ndarray          # IF 模型判定
    ae_predictions: np.ndarray          # AE 模型判定


class EnsembleDetector:
    """集成决策器

    融合 Isolation Forest（快速粗筛）和 Autoencoder（深度细检）
    的检测结果，支持三种集成策略：

    1. any: 任一模型判异常则异常（高召回率，可能有误报）
    2. majority: 多数模型判异常则异常（平衡策略）
    3. weighted: 加权融合异常分数（推荐，精度最高）
    """

    def __init__(
        self,
        if_detector: IsolationForestDetector,
        ae_detector: AutoencoderDetector,
        strategy: str = "weighted",
        if_weight: float = 0.4,
        ae_weight: float = 0.6,
    ) -> None:
        """初始化集成决策器

        Args:
            if_detector: Isolation Forest 检测器
            ae_detector: Autoencoder 检测器
            strategy: 集成策略 (any/majority/weighted)
            if_weight: IF 模型权重（weighted 策略）
            ae_weight: AE 模型权重（weighted 策略）
        """
        self.if_detector = if_detector
        self.ae_detector = ae_detector
        self.strategy = strategy
        self.if_weight = if_weight
        self.ae_weight = ae_weight

        # 归一化权重
        total = self.if_weight + self.ae_weight
        if total > 0:
            self.if_weight /= total
            self.ae_weight /= total

    def detect(self, X: np.ndarray) -> EnsembleResult:
        """执行集成异常检测

        分别调用两个模型进行检测，然后按指定策略融合结果。

        Args:
            X: 特征矩阵，shape: [n_samples, n_features]

        Returns:
            EnsembleResult: 集成检测结果
        """
        # IF 模型检测
        if_is_anomaly, if_scores = self.if_detector.detect(X)

        # AE 模型检测
        ae_is_anomaly, ae_errors = self.ae_detector.detect(X)

        # 将 IF 分数归一化到 [0, 1]（决策函数值，正值=正常，负值=异常）
        if_normalized = self._normalize_if_scores(if_scores)

        # 将 AE 误差归一化到 [0, 1]
        ae_normalized = self._normalize_ae_scores(ae_errors)

        # 集成决策
        if self.strategy == "any":
            is_anomaly = if_is_anomaly | ae_is_anomaly
            scores = np.maximum(if_normalized, ae_normalized)

        elif self.strategy == "majority":
            # 两个模型都认为异常才算异常
            is_anomaly = if_is_anomaly & ae_is_anomaly
            scores = (if_normalized + ae_normalized) / 2

        elif self.strategy == "weighted":
            # 加权融合
            scores = self.if_weight * if_normalized + self.ae_weight * ae_normalized
            # 使用加权分数的中位数作为阈值
            threshold = np.median(scores) + 1.5 * np.std(scores)
            is_anomaly = scores > threshold

        else:
            raise ValueError(f"未知集成策略: {self.strategy}")

        # IF 预测标签
        if_predictions = np.where(if_is_anomaly, -1, 1)
        ae_predictions = np.where(ae_is_anomaly, -1, 1)

        result = EnsembleResult(
            is_anomaly=is_anomaly,
            anomaly_scores=scores,
            if_scores=if_scores,
            ae_scores=ae_errors,
            if_predictions=if_predictions,
            ae_predictions=ae_predictions,
        )

        anomaly_count = int(np.sum(is_anomaly))
        logger.info(
            f"集成检测完成: {len(is_anomaly)} 样本, "
            f"异常 {anomaly_count} ({anomaly_count/len(is_anomaly)*100:.1f}%), "
            f"策略: {self.strategy}"
        )

        return result

    @staticmethod
    def _normalize_if_scores(scores: np.ndarray) -> np.ndarray:
        """将 IF 决策函数分数归一化到 [0, 1]

        决策函数：正值=正常，负值=异常
        归一化后：0=正常，1=异常

        Args:
            scores: IF 决策函数值

        Returns:
            归一化异常分数
        """
        if len(scores) == 0:
            return scores

        # 反转并归一化：越负 → 越接近 1
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s == min_s:
            return np.zeros_like(scores)

        # 映射：正常(正分)→0，异常(负分)→1
        normalized = (max_s - scores) / (max_s - min_s)
        return np.clip(normalized, 0, 1)

    @staticmethod
    def _normalize_ae_scores(errors: np.ndarray) -> np.ndarray:
        """将 AE 重构误差归一化到 [0, 1]

        Args:
            errors: AE 重构误差

        Returns:
            归一化异常分数
        """
        if len(errors) == 0:
            return errors

        min_e, max_e = np.min(errors), np.max(errors)
        if max_e == min_e:
            return np.zeros_like(errors)

        normalized = (errors - min_e) / (max_e - min_e)
        return np.clip(normalized, 0, 1)
