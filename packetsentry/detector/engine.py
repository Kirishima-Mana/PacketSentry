"""
PacketSentry 检测引擎核心

协调数据采集、特征提取、模型推理和结果输出的完整检测流程。
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from packetsentry.utils.config import PacketSentryConfig, load_config
from packetsentry.utils.logger import logger
from packetsentry.collector.parser import ParsedPacket
from packetsentry.collector.sniffer import PacketSniffer
from packetsentry.features.extractor import FeatureExtractor, FeatureWindow
from packetsentry.features.encoder import FeatureEncoder
from packetsentry.models.isolation_forest import IsolationForestDetector
from packetsentry.models.autoencoder import AutoencoderDetector
from packetsentry.models.ensemble import EnsembleDetector, EnsembleResult


@dataclass
class DetectionReport:
    """检测报告

    封装一次完整检测的所有结果和元数据。
    """
    total_windows: int = 0              # 总时间窗口数
    anomaly_windows: int = 0            # 异常窗口数
    anomaly_ratio: float = 0.0          # 异常比例
    duration: float = 0.0               # 检测耗时
    windows: list[FeatureWindow] = None  # 所有窗口
    ensemble_result: Optional[EnsembleResult] = None

    def __post_init__(self):
        if self.windows is None:
            self.windows = []


class DetectionEngine:
    """检测引擎

    完整的端到端异常检测流程：
    1. 数据采集（PCAP 文件或实时嗅探）
    2. 特征提取（时间窗口 + 统计特征）
    3. 模型推理（IF + AE 集成检测）
    4. 结果输出（报告 + 告警）
    """

    def __init__(self, config: Optional[PacketSentryConfig] = None) -> None:
        """初始化检测引擎

        Args:
            config: 配置对象，None 则使用默认配置
        """
        self.config = config or load_config()

        # 初始化各组件
        self.feature_extractor = FeatureExtractor(
            window_size=self.config.features.window_size,
            window_step=self.config.features.window_step,
        )
        self.feature_encoder = FeatureEncoder()
        self.if_detector = IsolationForestDetector(
            n_estimators=self.config.isolation_forest.n_estimators,
            contamination=self.config.isolation_forest.contamination,
            random_state=self.config.isolation_forest.random_state,
        )
        self.ae_detector = AutoencoderDetector(
            hidden_dims=self.config.autoencoder.hidden_dims,
            learning_rate=self.config.autoencoder.learning_rate,
            epochs=self.config.autoencoder.epochs,
            batch_size=self.config.autoencoder.batch_size,
            threshold_percentile=self.config.autoencoder.threshold_percentile,
        )
        self.ensemble = EnsembleDetector(
            if_detector=self.if_detector,
            ae_detector=self.ae_detector,
            strategy=self.config.detector.ensemble_strategy,
            if_weight=self.config.detector.if_weight,
            ae_weight=self.config.detector.ae_weight,
        )

        self._is_trained = False

    def train(self, packets: list[ParsedPacket]) -> None:
        """训练检测模型

        使用正常流量数据训练 IF 和 AE 模型。

        Args:
            packets: 正常流量数据包列表
        """
        logger.info(f"开始模型训练: {len(packets)} 个数据包")

        # 特征提取
        windows, feature_dicts = self.feature_extractor.extract_feature_matrix(packets)

        if not feature_dicts:
            logger.error("特征提取结果为空，无法训练")
            return

        # 编码为特征矩阵
        X = self.feature_encoder.encode_batch(feature_dicts)
        logger.info(f"特征矩阵: {X.shape}")

        # 拟合归一化参数
        self.feature_encoder.fit_normalize(X)

        # 训练 IF 模型
        self.if_detector.fit(X)

        # 训练 AE 模型
        self.ae_detector.fit(X)

        self._is_trained = True
        logger.info("模型训练完成")

    def train_from_pcap(self, pcap_path: str) -> None:
        """从 PCAP 文件训练模型

        Args:
            pcap_path: PCAP 文件路径
        """
        sniffer = PacketSniffer()
        packets = sniffer.read_pcap(pcap_path)
        self.train(packets)

    def detect(self, packets: list[ParsedPacket]) -> DetectionReport:
        """对数据包执行异常检测

        Args:
            packets: 待检测数据包列表

        Returns:
            DetectionReport: 检测报告
        """
        start_time = time.time()

        report = DetectionReport()

        # 特征提取
        windows, feature_dicts = self.feature_extractor.extract_feature_matrix(packets)
        report.windows = windows

        if not feature_dicts:
            logger.warning("无有效特征，跳过检测")
            return report

        # 编码
        X = self.feature_encoder.encode_batch(feature_dicts)
        report.total_windows = len(windows)

        if not self._is_trained:
            logger.error("模型未训练，请先调用 train() 方法")
            return report

        # 集成检测
        result = self.ensemble.detect(X)
        report.ensemble_result = result

        # 标记异常窗口
        anomaly_count = 0
        for i, window in enumerate(windows):
            if i < len(result.is_anomaly) and result.is_anomaly[i]:
                window.is_anomaly = True
                anomaly_count += 1

        report.anomaly_windows = anomaly_count
        report.anomaly_ratio = anomaly_count / max(report.total_windows, 1)
        report.duration = time.time() - start_time

        logger.info(
            f"检测完成: {report.total_windows} 窗口, "
            f"异常 {anomaly_count} ({report.anomaly_ratio*100:.1f}%), "
            f"耗时 {report.duration:.2f}s"
        )

        return report

    def detect_pcap(self, pcap_path: str) -> DetectionReport:
        """对 PCAP 文件执行异常检测

        Args:
            pcap_path: PCAP 文件路径

        Returns:
            DetectionReport
        """
        sniffer = PacketSniffer()
        packets = sniffer.read_pcap(pcap_path)
        return self.detect(packets)

    def save_models(self, directory: str) -> None:
        """保存所有训练好的模型

        Args:
            directory: 保存目录
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.if_detector.save(str(Path(directory) / "isolation_forest.pkl"))
        self.ae_detector.save(str(Path(directory) / "autoencoder.pkl"))
        logger.info(f"模型已保存到: {directory}")

    def load_models(self, directory: str) -> None:
        """加载已训练的模型

        Args:
            directory: 模型目录
        """
        self.if_detector.load(str(Path(directory) / "isolation_forest.pkl"))
        self.ae_detector.load(str(Path(directory) / "autoencoder.pkl"))
        self._is_trained = True
        logger.info(f"模型已从 {directory} 加载")

    @property
    def is_trained(self) -> bool:
        """模型是否已训练"""
        return self._is_trained
