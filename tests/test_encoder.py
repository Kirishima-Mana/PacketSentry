"""
PacketSentry 单元测试 - 特征编码器
"""

import pytest
import numpy as np
import pandas as pd
from packetsentry.features.encoder import FeatureEncoder, FEATURE_COLUMNS


class TestFeatureEncoder:
    """特征编码器测试"""

    def test_feature_columns(self):
        """测试特征列名"""
        assert len(FEATURE_COLUMNS) > 20  # 应有足够多的特征
        assert "total_packets" in FEATURE_COLUMNS
        assert "ratio_tcp" in FEATURE_COLUMNS
        assert "unique_src_ip" in FEATURE_COLUMNS

    def test_encode_single(self):
        """测试单个特征字典编码"""
        encoder = FeatureEncoder()
        feature_dict = {
            "total_packets": 100.0,
            "total_bytes": 150000.0,
            "ratio_tcp": 0.8,
            "unique_src_ip": 10.0,
        }
        vector = encoder.encode(feature_dict)
        assert vector.shape == (len(FEATURE_COLUMNS),)
        assert vector.dtype == np.float64

        # 检查特定特征值
        total_packets_idx = FEATURE_COLUMNS.index("total_packets")
        assert vector[total_packets_idx] == 100.0

        # 检查缺失特征填充为0
        ratio_udp_idx = FEATURE_COLUMNS.index("ratio_udp")
        assert vector[ratio_udp_idx] == 0.0

    def test_encode_batch(self):
        """测试批量编码"""
        encoder = FeatureEncoder()
        feature_dicts = [
            {"total_packets": 100.0, "total_bytes": 150000.0},
            {"total_packets": 200.0, "total_bytes": 300000.0},
            {"total_packets": 50.0, "total_bytes": 75000.0},
        ]
        matrix = encoder.encode_batch(feature_dicts)
        assert matrix.shape == (3, len(FEATURE_COLUMNS))
        assert matrix.dtype == np.float64

        total_packets_idx = FEATURE_COLUMNS.index("total_packets")
        assert matrix[0, total_packets_idx] == 100.0
        assert matrix[1, total_packets_idx] == 200.0
        assert matrix[2, total_packets_idx] == 50.0

    def test_encode_empty_batch(self):
        """测试空批量编码"""
        encoder = FeatureEncoder()
        matrix = encoder.encode_batch([])
        assert matrix.shape == (0, len(FEATURE_COLUMNS))

    def test_fit_normalize(self):
        """测试归一化参数拟合"""
        encoder = FeatureEncoder()
        # 创建测试数据
        n_samples = 100
        n_features = len(FEATURE_COLUMNS)
        X = np.random.randn(n_samples, n_features)

        encoder.fit_normalize(X)
        assert encoder._means is not None
        assert encoder._stds is not None
        assert encoder._means.shape == (n_features,)
        assert encoder._stds.shape == (n_features,)
        assert np.all(encoder._stds > 0)  # 标准差应为正

    def test_normalize(self):
        """测试 Z-Score 归一化"""
        encoder = FeatureEncoder()
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        encoder.fit_normalize(X)
        X_norm = encoder.normalize(X)

        # 归一化后每列均值为0，标准差为1（使用 ddof=0 与 sklearn 一致）
        assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0, ddof=0), 1.0, atol=1e-10)

    def test_normalize_without_fit(self):
        """测试未拟合时归一化报错"""
        encoder = FeatureEncoder()
        X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="请先调用 fit_normalize"):
            encoder.normalize(X)

    def test_to_dataframe(self):
        """测试转换为 DataFrame"""
        encoder = FeatureEncoder()
        feature_dicts = [
            {"total_packets": 100.0, "total_bytes": 150000.0},
            {"total_packets": 200.0, "total_bytes": 300000.0},
        ]
        df = encoder.to_dataframe(feature_dicts)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, len(FEATURE_COLUMNS))
        assert "total_packets" in df.columns
        assert "total_bytes" in df.columns
        assert df["total_packets"].iloc[0] == 100.0
        assert df["total_packets"].iloc[1] == 200.0

    def test_encode_with_nan_inf(self):
        """测试处理 NaN 和 Inf 值"""
        encoder = FeatureEncoder()
        feature_dict = {
            "total_packets": float("nan"),
            "total_bytes": float("inf"),
            "ratio_tcp": -float("inf"),
        }
        vector = encoder.encode(feature_dict)
        # NaN 和 Inf 应被替换为 0.0
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0.0)
