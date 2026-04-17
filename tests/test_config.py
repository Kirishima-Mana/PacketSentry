"""
PacketSentry 单元测试 - 数据模型与配置
"""

import pytest
from packetsentry.utils.config import (
    PacketSentryConfig,
    CollectorConfig,
    FeatureConfig,
    IsolationForestConfig,
    AutoencoderConfig,
    DetectorConfig,
    load_config,
)


class TestConfigClasses:
    """配置类测试"""

    def test_collector_config_defaults(self):
        """测试 CollectorConfig 默认值"""
        cfg = CollectorConfig()
        assert cfg.interface == "eth0"
        assert cfg.bpf_filter == ""
        assert cfg.timeout == 60
        assert cfg.max_packets == 0

    def test_feature_config_defaults(self):
        """测试 FeatureConfig 默认值"""
        cfg = FeatureConfig()
        assert cfg.window_size == 5.0
        assert cfg.window_step == 2.5
        assert cfg.deep_protocol_analysis is True

    def test_isolation_forest_config_defaults(self):
        """测试 IsolationForestConfig 默认值"""
        cfg = IsolationForestConfig()
        assert cfg.n_estimators == 200
        assert cfg.contamination == 0.05
        assert cfg.random_state == 42
        assert cfg.max_features == 1.0

    def test_autoencoder_config_defaults(self):
        """测试 AutoencoderConfig 默认值"""
        cfg = AutoencoderConfig()
        assert cfg.hidden_dims == [32, 16, 8]
        assert cfg.learning_rate == 0.001
        assert cfg.epochs == 100
        assert cfg.batch_size == 32
        assert cfg.threshold_percentile == 95.0

    def test_detector_config_defaults(self):
        """测试 DetectorConfig 默认值"""
        cfg = DetectorConfig()
        assert cfg.ensemble_strategy == "weighted"
        assert cfg.if_weight == 0.4
        assert cfg.ae_weight == 0.6
        assert cfg.alert_cooldown == 30

    def test_packetsentry_config_defaults(self):
        """测试 PacketSentryConfig 默认值"""
        cfg = PacketSentryConfig()
        assert isinstance(cfg.collector, CollectorConfig)
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.isolation_forest, IsolationForestConfig)
        assert isinstance(cfg.autoencoder, AutoencoderConfig)
        assert isinstance(cfg.detector, DetectorConfig)


class TestConfigLoading:
    """配置加载测试"""

    def test_load_default_config(self):
        """测试加载默认配置"""
        cfg = load_config()
        assert isinstance(cfg, PacketSentryConfig)
        assert cfg.collector.interface == "eth0"
        assert cfg.features.window_size == 5.0
        assert cfg.isolation_forest.n_estimators == 200
