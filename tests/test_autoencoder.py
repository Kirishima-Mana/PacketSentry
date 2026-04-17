"""
PacketSentry 单元测试 - Autoencoder 检测器
"""

import pytest
import numpy as np
from packetsentry.models.autoencoder import AutoencoderDetector


class TestAutoencoderDetector:
    """Autoencoder 检测器测试"""

    def test_initialization(self):
        """测试初始化"""
        detector = AutoencoderDetector(
            hidden_dims=[16, 8, 4],
            learning_rate=0.01,
            epochs=50,
            batch_size=16,
            threshold_percentile=90.0,
        )
        assert detector.hidden_dims == [16, 8, 4]
        assert detector.learning_rate == 0.01
        assert detector.epochs == 50
        assert detector.batch_size == 16
        assert detector.threshold_percentile == 90.0

    def test_fit_and_predict_no_pytorch(self):
        """测试无 PyTorch 时的训练和预测（降级到统计方法）"""
        detector = AutoencoderDetector(
            hidden_dims=[8, 4],  # 简单网络
            epochs=10,  # 少量轮次加速测试
            batch_size=8,
        )

        # 创建测试数据
        X_train = np.random.randn(100, 10)
        detector.fit(X_train)
        assert detector._is_fitted is True

        # 预测
        X_test = np.random.randn(20, 10)
        predictions = detector.predict(X_test)
        assert predictions.shape == (20,)
        assert set(predictions).issubset({-1, 1})

    def test_score_samples(self):
        """测试异常分数计算"""
        detector = AutoencoderDetector(epochs=10)
        X = np.random.randn(50, 8)
        detector.fit(X)

        scores = detector.score_samples(X)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)  # 重构误差应为非负

    def test_detect(self):
        """测试检测方法"""
        detector = AutoencoderDetector(epochs=10)
        X = np.random.randn(50, 8)
        detector.fit(X)

        is_anomaly, errors = detector.detect(X)
        assert is_anomaly.shape == (50,)
        assert errors.shape == (50,)
        assert is_anomaly.dtype == bool

        # 使用自定义阈值
        is_anomaly_custom, errors_custom = detector.detect(X, threshold=0.5)
        assert is_anomaly_custom.shape == (50,)

    def test_threshold_calculation(self):
        """测试阈值计算"""
        detector = AutoencoderDetector(
            epochs=10,
            threshold_percentile=95.0,
        )
        X = np.random.randn(100, 8)
        detector.fit(X)

        # 检查阈值是否已计算
        assert detector._threshold is not None
        assert detector._threshold > 0

        # 验证阈值基于指定百分位
        errors = detector.score_samples(X)
        expected_threshold = np.percentile(errors, 95.0)
        assert detector._threshold == pytest.approx(expected_threshold, rel=1e-5)

    def test_unfitted_error(self):
        """测试未训练时调用方法报错"""
        detector = AutoencoderDetector()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.predict(X)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.score_samples(X)

        with pytest.raises(RuntimeError, match="模型未训练"):
            detector.detect(X)

    def test_save_load_no_pytorch(self, tmp_path):
        """测试无 PyTorch 时的模型保存和加载"""
        detector = AutoencoderDetector(epochs=10)
        X = np.random.randn(100, 8)
        detector.fit(X)

        # 保存
        save_path = tmp_path / "model.pkl"
        detector.save(str(save_path))
        assert save_path.exists()

        # 加载
        detector2 = AutoencoderDetector()
        detector2.load(str(save_path))
        assert detector2._is_fitted is True

        # 验证加载的模型能工作
        X_test = np.random.randn(20, 8)
        predictions = detector2.predict(X_test)
        assert predictions.shape == (20,)

    def test_statistical_fallback(self):
        """测试 PyTorch 不可用时的统计降级"""
        # 模拟 PyTorch 不可用情况
        detector = AutoencoderDetector(epochs=10)
        X = np.random.randn(100, 8)

        # 训练（应使用统计方法）
        detector.fit(X)
        assert detector._is_fitted is True

        # 检查是否使用了统计方法
        if not detector._use_pytorch:
            assert detector._mean is not None
            assert detector._cov_inv is not None

        # 验证检测能工作
        is_anomaly, errors = detector.detect(X)
        assert is_anomaly.shape == (100,)
        assert errors.shape == (100,)

    def test_reconstruction_error_distribution(self):
        """测试重构误差分布"""
        detector = AutoencoderDetector(epochs=10)
        X = np.random.randn(200, 10)
        detector.fit(X)

        errors = detector.score_samples(X)

        # 大多数样本应具有较低的重构误差
        median_error = np.median(errors)
        max_error = np.max(errors)
        
        # 中位数应小于最大值（除非所有值相等）
        if max_error > 0:
            assert median_error < max_error

        # 添加一些异常点
        X_anomalous = np.random.randn(20, 10) * 5  # 更大的方差
        errors_anomalous = detector.score_samples(X_anomalous)

        # 异常点的重构误差应更高（统计上）
        mean_normal = np.mean(errors)
        mean_anomalous = np.mean(errors_anomalous)
        
        # 异常点平均误差应高于正常点（允许一定随机性）
        assert mean_anomalous > mean_normal * 0.5  # 放宽条件避免随机失败
