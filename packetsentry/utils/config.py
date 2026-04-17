"""
PacketSentry 配置管理模块

加载和管理系统配置，支持 INI 格式配置文件和环境变量覆盖。
"""

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.ini"


@dataclass
class CollectorConfig:
    """数据采集配置"""
    interface: str = "eth0"               # 嗅探网卡接口
    bpf_filter: str = ""                  # BPF 过滤器表达式
    timeout: int = 60                     # 抓包超时（秒）
    max_packets: int = 0                  # 最大包数量（0=无限）


@dataclass
class FeatureConfig:
    """特征工程配置"""
    window_size: float = 5.0              # 时间窗口大小（秒）
    window_step: float = 2.5              # 窗口滑动步长（秒）
    deep_protocol_analysis: bool = True   # 是否启用协议深度解析


@dataclass
class IsolationForestConfig:
    """Isolation Forest 模型配置"""
    n_estimators: int = 200               # 树数量
    contamination: float = 0.05           # 异常比例假设
    random_state: int = 42                # 随机种子
    max_features: float = 1.0             # 特征采样比例


@dataclass
class AutoencoderConfig:
    """Autoencoder 模型配置"""
    hidden_dims: list[int] = field(       # 编码层维度
        default_factory=lambda: [32, 16, 8]
    )
    learning_rate: float = 0.001          # 学习率
    epochs: int = 100                     # 训练轮次
    batch_size: int = 32                  # 批大小
    threshold_percentile: float = 95.0    # 异常阈值百分位


@dataclass
class DetectorConfig:
    """检测引擎配置"""
    ensemble_strategy: str = "weighted"   # 集成策略：any/majority/weighted
    if_weight: float = 0.4               # IF 权重
    ae_weight: float = 0.6               # AE 权重
    alert_cooldown: int = 30              # 告警冷却时间（秒）


@dataclass
class PacketSentryConfig:
    """PacketSentry 完整配置"""
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


def load_config(config_path: Optional[str] = None) -> PacketSentryConfig:
    """加载配置文件

    优先级：指定配置文件 > 环境变量 > 默认配置

    Args:
        config_path: 配置文件路径，None 则使用默认配置

    Returns:
        PacketSentryConfig: 完整配置对象
    """
    config = PacketSentryConfig()

    # 确定配置文件路径
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return config

    # 解析 INI 文件
    parser = configparser.ConfigParser()
    parser.read(str(path), encoding="utf-8")

    # 采集配置
    if parser.has_section("collector"):
        sec = parser["collector"]
        config.collector.interface = sec.get("interface", config.collector.interface)
        config.collector.bpf_filter = sec.get("bpf_filter", config.collector.bpf_filter)
        config.collector.timeout = sec.getint("timeout", config.collector.timeout)
        config.collector.max_packets = sec.getint("max_packets", config.collector.max_packets)

    # 特征配置
    if parser.has_section("features"):
        sec = parser["features"]
        config.features.window_size = sec.getfloat("window_size", config.features.window_size)
        config.features.window_step = sec.getfloat("window_step", config.features.window_step)
        config.features.deep_protocol_analysis = sec.getboolean(
            "deep_protocol_analysis", config.features.deep_protocol_analysis
        )

    # IF 模型配置
    if parser.has_section("models.isolation_forest"):
        sec = parser["models.isolation_forest"]
        config.isolation_forest.n_estimators = sec.getint(
            "n_estimators", config.isolation_forest.n_estimators
        )
        config.isolation_forest.contamination = sec.getfloat(
            "contamination", config.isolation_forest.contamination
        )
        config.isolation_forest.random_state = sec.getint(
            "random_state", config.isolation_forest.random_state
        )

    # AE 模型配置
    if parser.has_section("models.autoencoder"):
        sec = parser["models.autoencoder"]
        dims_str = sec.get("hidden_dims", "32,16,8")
        config.autoencoder.hidden_dims = [int(x.strip()) for x in dims_str.split(",")]
        config.autoencoder.learning_rate = sec.getfloat(
            "learning_rate", config.autoencoder.learning_rate
        )
        config.autoencoder.epochs = sec.getint("epochs", config.autoencoder.epochs)
        config.autoencoder.batch_size = sec.getint("batch_size", config.autoencoder.batch_size)
        config.autoencoder.threshold_percentile = sec.getfloat(
            "threshold_percentile", config.autoencoder.threshold_percentile
        )

    # 检测配置
    if parser.has_section("detector"):
        sec = parser["detector"]
        config.detector.ensemble_strategy = sec.get(
            "ensemble_strategy", config.detector.ensemble_strategy
        )
        config.detector.if_weight = sec.getfloat("if_weight", config.detector.if_weight)
        config.detector.ae_weight = sec.getfloat("ae_weight", config.detector.ae_weight)
        config.detector.alert_cooldown = sec.getint(
            "alert_cooldown", config.detector.alert_cooldown
        )

    return config
