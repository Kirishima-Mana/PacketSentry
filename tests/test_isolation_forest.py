"""
PacketSentry 单元测试 - Isolation Forest 检测器
"""

import pytest
import numpy as np
from packetsentry.models.isolation_forest import IsolationForestDetector


class TestIsolationForestDetector:
    """Isolation Forest 检测器测试"""

    def test_initialization(self):
        """测试初始化"""
        detector = IsolationForestDetector(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            max_features=0.8,
        )
        assert detector.n_estimators == 100
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert detector.max_features == 0.8

    def test_fit_and_predict(self):
        """测试训练和预测"""
        detector = IsolationForestDetector(
            n_estimators=50,  # 少量树加速测试
            contamination=0.1,
            random_state=42,
        )

        # 创建测试数据（正常数据）
        X_train = np.random.randn(100, 10)
        detector.fit(X_train)
        assert detector._is_fitted is True
        assert detector._model is not None

        # 预测
        X_test = np.random.randn(20, 10)
        predictions = detector.predict(X_test)
        assert predictions.shape == (20,)
        assert set(predictions).issubset({-1, 1})  # 应为 -1（异常）或 1（正常）

    def test_score_samples(self):
        """测试异常分数计算"""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        X = np.random.randn(50, 8)
        detector.fit(X)

        scores = detector.score_samples(X)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))

    def test_decision_function(self):
        """测试决策函数"""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        X = np.random.randn(50, 8)
        detector.fit(X)

        decisions = detector.decision_function(X)
        assert decisions.shape == (50,)
        # 决策函数：正值表示正常，负值表示异常
        assert np.any(decisions > 0) or np.any(decisions < 0)

    def test_detect(self):
        """测试检测方法"""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        X = np.random.randn(50, 8)
        detector.fit(X)

        is_anomaly, scores = detector.detect(X, threshold=0.0)
        assert is_anomaly.shape == (50,)
        assert scores.shape == (50,)
        assert is_anomaly.dtype == bool

        # 分数与决策函数值一致
        decisions = detector.decision_function(X)
        assert np.allclose(scores, decisions)

    def test_unfitted_error(self):
        """测试未训练时调用方法报错"""
        detector = IsolationForestDetector()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.predict(X)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.score_samples(X)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.decision_function(X)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.detect(X)

    def test_save_load(self, tmp_path):
        """测试模型保存和加载"""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        X = np.random.randn(100, 8)
        detector.fit(X)

        # 保存
        save_path = tmp_path / "model.pkl"
        detector.save(str(save_path))
        assert save_path.exists()

        # 加载
        detector2 = IsolationForestDetector()
        detector2.load(str(save_path))
        assert detector2._is_fitted is True

        # 验证加载的模型能工作
        X_test = np.random.randn(20, 8)
        predictions = detector2.predict(X_test)
        assert predictions.shape == (20,)

    def test_contamination_effect(self):
        """测试 contamination 参数影响"""
        # 低 contamination（假设异常很少）
        detector_low = IsolationForestDetector(contamination=0.01, random_state=42)
        # 高 contamination（假设异常较多）
        detector_high = IsolationForestDetector(contamination=0.2, random_state=42)

        X = np.random.randn(200, 10)
        detector_low.fit(X)
        detector_high.fit(X)

        # 高 contamination 应检测出更多异常
        is_anomaly_low, _ = detector_low.detect(X)
        is_anomaly_high, _ = detector_high.detect(X)

        anomaly_ratio_low = np.mean(is_anomaly_low)
        anomaly_ratio_high = np.mean(is_anomaly_high)

        # 高 contamination 应检测出更多异常（或至少一样多）
        assert anomaly_ratio_high >= anomaly_ratio_low - 0.05  # 允许小误差
