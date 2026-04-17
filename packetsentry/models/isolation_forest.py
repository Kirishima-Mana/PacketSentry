"""
PacketSentry Isolation Forest 异常检测模型

基于 scikit-learn 的 Isolation Forest 算法实现，
适用于无监督条件下的快速异常检测。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from packetsentry.utils.logger import logger


class IsolationForestDetector:
    """Isolation Forest 异常检测器

    利用 Isolation Forest 算法对流量特征向量进行异常评分。
    该算法通过随机分割特征空间来隔离异常点，异常点通常
    需要更少的分割次数即可被隔离，因此路径长度更短。

    优势：
    - 不需要标注数据（无监督）
    - 训练速度快，适合实时检测
    - 对高维特征空间有良好表现
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        max_features: float = 1.0,
    ) -> None:
        """初始化 IF 检测器

        Args:
            n_estimators: 树数量，越多越精确但越慢
            contamination: 异常比例假设，影响阈值
            random_state: 随机种子
            max_features: 特征采样比例
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.max_features = max_features

        self._model: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """训练模型

        使用正常流量特征训练 Isolation Forest。
        注意：contamination 参数应设置为训练数据中
        预期的异常比例（通常很小）。

        Args:
            X: 训练特征矩阵，shape: [n_samples, n_features]

        Returns:
            self（支持链式调用）
        """
        logger.info(
            f"训练 Isolation Forest: {X.shape[0]} 样本, "
            f"{X.shape[1]} 特征, n_estimators={self.n_estimators}"
        )

        # 标准化特征
        X_scaled = self._scaler.fit_transform(X)

        # 训练模型
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,  # 并行训练
        )
        self._model.fit(X_scaled)
        self._is_fitted = True

        logger.info("Isolation Forest 训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本是否为异常

        Args:
            X: 待预测特征矩阵

        Returns:
            预测标签数组，1=正常，-1=异常

        Raises:
            RuntimeError: 模型未训练
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数

        分数越低越异常。基于平均路径长度，
        正常样本的路径更长，分数更高。

        Args:
            X: 特征矩阵

        Returns:
            异常分数数组，越低越异常
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.score_samples(X_scaled)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值

        正值表示正常，负值表示异常。

        Args:
            X: 特征矩阵

        Returns:
            决策函数值数组
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)

    def detect(self, X: np.ndarray, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """检测异常并返回结果

        Args:
            X: 特征矩阵
            threshold: 决策函数阈值，低于此值判定为异常

        Returns:
            (是否异常布尔数组, 异常分数数组)
        """
        scores = self.decision_function(X)
        is_anomaly = scores < threshold
        return is_anomaly, scores

    def save(self, path: str) -> None:
        """保存模型到文件

        Args:
            path: 保存路径
        """
        self._check_fitted()
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "max_features": self.max_features,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str) -> "IsolationForestDetector":
        """从文件加载模型

        Args:
            path: 模型文件路径

        Returns:
            self
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._scaler = data["scaler"]
        self._is_fitted = True
        logger.info(f"模型已加载: {path}")
        return self

    def _check_fitted(self) -> None:
        """检查模型是否已训练"""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit() 方法")
