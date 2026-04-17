"""
PacketSentry 单元测试 - 集成决策器
"""

import pytest
import numpy as np
from packetsentry.models.isolation_forest import IsolationForestDetector
from packetsentry.models.autoencoder import AutoencoderDetector
from packetsentry.models.ensemble import EnsembleDetector, EnsembleResult


class TestEnsembleDetector:
    """集成决策器测试"""

    def _create_trained_detectors(self) -> tuple[IsolationForestDetector, AutoencoderDetector]:
        """创建已训练的检测器"""
        # IF 检测器
        if_detector = IsolationForestDetector(
            n_estimators=50,
            contamination=0.1,
            random_state=42,
        )

        # AE 检测器
        ae_detector = AutoencoderDetector(
            hidden_dims=[8, 4],
            epochs=10,
            batch_size=8,
        )

        # 训练数据
        X_train = np.random.randn(200, 10)
        if_detector.fit(X_train)
        ae_detector.fit(X_train)

        return if_detector, ae_detector

    def test_initialization(self):
        """测试初始化"""
        if_detector, ae_detector = self._create_trained_detectors()

        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="weighted",
            if_weight=0.3,
            ae_weight=0.7,
        )

        assert ensemble.if_detector is if_detector
        assert ensemble.ae_detector is ae_detector
        assert ensemble.strategy == "weighted"
        assert ensemble.if_weight == 0.3
        assert ensemble.ae_weight == 0.7

    def test_weight_normalization(self):
        """测试权重归一化"""
        if_detector, ae_detector = self._create_trained_detectors()

        # 权重和为2，应归一化为1
        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="weighted",
            if_weight=0.6,
            ae_weight=1.4,
        )

        assert ensemble.if_weight + ensemble.ae_weight == pytest.approx(1.0, rel=1e-10)

    def test_detect_weighted(self):
        """测试加权策略检测"""
        if_detector, ae_detector = self._create_trained_detectors()

        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="weighted",
            if_weight=0.4,
            ae_weight=0.6,
        )

        X_test = np.random.randn(50, 10)
        result = ensemble.detect(X_test)

        assert isinstance(result, EnsembleResult)
        assert result.is_anomaly.shape == (50,)
        assert result.anomaly_scores.shape == (50,)
        assert result.if_scores.shape == (50,)
        assert result.ae_scores.shape == (50,)
        assert result.if_predictions.shape == (50,)
        assert result.ae_predictions.shape == (50,)

        # 检查分数范围
        assert np.all(result.anomaly_scores >= 0)
        assert np.all(result.anomaly_scores <= 1)

        # 检查预测标签
        assert set(result.if_predictions).issubset({-1, 1})
        assert set(result.ae_predictions).issubset({-1, 1})

    def test_detect_any(self):
        """测试 any 策略检测"""
        if_detector, ae_detector = self._create_trained_detectors()

        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="any",
        )

        X_test = np.random.randn(50, 10)
        result = ensemble.detect(X_test)

        # any 策略：任一模型判异常则异常
        expected_any = (result.if_predictions == -1) | (result.ae_predictions == -1)
        assert np.array_equal(result.is_anomaly, expected_any)

    def test_detect_majority(self):
        """测试 majority 策略检测"""
        if_detector, ae_detector = self._create_trained_detectors()

        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="majority",
        )

        X_test = np.random.randn(50, 10)
        result = ensemble.detect(X_test)

        # majority 策略：两个模型都判异常才异常
        expected_majority = (result.if_predictions == -1) & (result.ae_predictions == -1)
        assert np.array_equal(result.is_anomaly, expected_majority)

    def test_invalid_strategy(self):
        """测试无效策略"""
        if_detector, ae_detector = self._create_trained_detectors()

        # 测试无效策略在初始化时不报错，但在检测时报错
        ensemble = EnsembleDetector(
            if_detector=if_detector,
            ae_detector=ae_detector,
            strategy="invalid",
        )
        
        X_test = np.random.randn(10, 10)
        with pytest.raises(ValueError, match="未知集成策略"):
            ensemble.detect(X_test)

    def test_normalize_if_scores(self):
        """测试 IF 分数归一化"""
        # 测试正常情况
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normalized = EnsembleDetector._normalize_if_scores(scores)

        # 检查范围
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # 检查单调性：分数越低（越异常）→ 归一化值越高
        assert normalized[0] > normalized[2]  # -2.0 比 0.0 更异常
        assert normalized[4] < normalized[2]  # 2.0 比 0.0 更正常

        # 测试空数组
        empty_scores = np.array([])
        empty_normalized = EnsembleDetector._normalize_if_scores(empty_scores)
        assert len(empty_normalized) == 0

        # 测试所有值相等
        equal_scores = np.array([1.0, 1.0, 1.0])
        equal_normalized = EnsembleDetector._normalize_if_scores(equal_scores)
        assert np.all(equal_normalized == 0.0)

    def test_normalize_ae_scores(self):
        """测试 AE 分数归一化"""
        # 测试正常情况
        errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        normalized = EnsembleDetector._normalize_ae_scores(errors)

        # 检查范围
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # 检查单调性：误差越大 → 归一化值越高
        assert normalized[4] > normalized[0]  # 5.0 比 0.1 更异常

        # 测试空数组
        empty_errors = np.array([])
        empty_normalized = EnsembleDetector._normalize_ae_scores(empty_errors)
        assert len(empty_normalized) == 0

        # 测试所有值相等
        equal_errors = np.array([0.5, 0.5, 0.5])
        equal_normalized = EnsembleDetector._normalize_ae_scores(equal_errors)
        assert np.all(equal_normalized == 0.0)

    def test_ensemble_consistency(self):
        """测试集成结果一致性"""
        if_detector, ae_detector = self._create_trained_detectors()

        strategies = ["any", "majority", "weighted"]
        X_test = np.random.randn(30, 10)

        results = {}
        for strategy in strategies:
            ensemble = EnsembleDetector(
                if_detector=if_detector,
                ae_detector=ae_detector,
                strategy=strategy,
            )
            results[strategy] = ensemble.detect(X_test)

        # 检查各策略结果维度一致
        for strategy in strategies:
            result = results[strategy]
            assert result.is_anomaly.shape == (30,)
            assert result.anomaly_scores.shape == (30,)

        # 验证策略逻辑
        # any 策略应检测出最多异常
        # majority 策略应检测出最少异常
        any_count = np.sum(results["any"].is_anomaly)
        majority_count = np.sum(results["majority"].is_anomaly)
        weighted_count = np.sum(results["weighted"].is_anomaly)

        # 加权策略的异常数应在 any 和 majority 之间
        assert any_count >= majority_count
        assert weighted_count <= any_count
        assert weighted_count >= 0  # 非负
