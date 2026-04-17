"""
Autoencoder 异常检测器

基于 PyTorch 的自动编码器，用于无监督异常检测。
当 PyTorch 不可用时，降级到基于马氏距离的统计检测器。
"""

import pickle
from typing import Optional

import numpy as np
from sklearn.covariance import MinCovDet

from packetsentry.utils.logger import logger


class AutoencoderDetector:
    """Autoencoder 异常检测器

    使用自动编码器重构误差作为异常分数。
    当 PyTorch 不可用时，使用马氏距离作为替代。
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        threshold_percentile: float = 95.0,
    ) -> None:
        """初始化 Autoencoder 检测器

        Args:
            hidden_dims: 编码层维度列表，例如 [32, 16, 8]
            learning_rate: 学习率
            epochs: 训练轮次
            batch_size: 批大小
            threshold_percentile: 异常阈值百分位
        """
        self.hidden_dims = hidden_dims or [32, 16, 8]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile

        # 模型状态
        self._is_fitted = False
        self._threshold = None
        self._use_pytorch = False
        self._model = None
        self._mean = None
        self._cov_inv = None

        # 检查 PyTorch 可用性
        try:
            import torch
            import torch.nn as nn
            self._use_pytorch = True
        except ImportError:
            logger.warning("PyTorch 未安装，使用基于马氏距离的简化检测器")
            self._use_pytorch = False

    def fit(self, X: np.ndarray) -> None:
        """训练 Autoencoder 检测器

        Args:
            X: 训练特征矩阵 (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        logger.info(f"训练 Autoencoder: {n_samples} 样本, {n_features} 特征, epochs={self.epochs}")

        if self._use_pytorch:
            self._fit_pytorch(X)
        else:
            self._fit_statistical(X)

        # 计算异常阈值（直接计算，不调用 score_samples）
        if self._use_pytorch:
            scores = self._compute_reconstruction_errors(X)
        else:
            # 统计方法：马氏距离
            X_centered = X - self._mean
            scores = np.sum(X_centered @ self._cov_inv * X_centered, axis=1)
        
        self._threshold = np.percentile(scores, self.threshold_percentile)
        logger.info(f"异常阈值: {self._threshold:.6f} (百分位: {self.threshold_percentile})")

        self._is_fitted = True

    def _fit_pytorch(self, X: np.ndarray) -> None:
        """使用 PyTorch 训练 Autoencoder"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # 定义 Autoencoder 网络
        class Autoencoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int]):
                super().__init__()
                # 编码器
                encoder_layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    encoder_layers.append(nn.Linear(prev_dim, h_dim))
                    encoder_layers.append(nn.ReLU())
                    prev_dim = h_dim

                # 解码器（对称结构）
                decoder_layers = []
                for h_dim in reversed(hidden_dims[:-1]):
                    decoder_layers.append(nn.Linear(prev_dim, h_dim))
                    decoder_layers.append(nn.ReLU())
                    prev_dim = h_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))

                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X)

        # 创建模型
        self._model = Autoencoder(X.shape[1], self.hidden_dims)
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # 训练循环
        n_batches = int(np.ceil(len(X) / self.batch_size))
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i in range(n_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, len(X))
                batch = X_tensor[start:end]

                optimizer.zero_grad()
                reconstructed = self._model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.debug(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.6f}")

    def _fit_statistical(self, X: np.ndarray) -> None:
        """使用统计方法训练（马氏距离）"""
        # 计算均值和协方差
        self._mean = np.mean(X, axis=0)
        
        # 使用最小协方差行列式估计器（对异常值更鲁棒）
        try:
            cov_estimator = MinCovDet().fit(X)
            cov = cov_estimator.covariance_
        except Exception:
            # 回退到样本协方差
            cov = np.cov(X, rowvar=False)
        
        # 添加小正则项确保可逆
        cov += np.eye(cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)
        
        logger.info("简化检测器（马氏距离）训练完成")

    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """计算重构误差（仅 PyTorch 模式）"""
        if not self._use_pytorch:
            raise RuntimeError("此方法仅适用于 PyTorch 模式")
        
        import torch
        
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstructed = self._model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return errors.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本标签

        Args:
            X: 特征矩阵

        Returns:
            预测标签数组，1=正常，-1=异常
        """
        is_anomaly, _ = self.detect(X)
        return np.where(is_anomaly, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数（重构误差）

        分数越高越异常。

        Args:
            X: 特征矩阵

        Returns:
            异常分数数组
        """
        if not self._is_fitted:
            raise RuntimeError("模型未训练")
        
        if self._use_pytorch:
            return self._compute_reconstruction_errors(X)
        else:
            # 统计方法：马氏距离
            X_centered = X - self._mean
            mahalanobis = np.sum(X_centered @ self._cov_inv * X_centered, axis=1)
            return mahalanobis

    def detect(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """检测异常

        Args:
            X: 特征矩阵
            threshold: 自定义阈值（None 时使用训练时计算的阈值）

        Returns:
            (is_anomaly, scores): 异常标志数组和异常分数数组
        """
        if not self._is_fitted:
            raise RuntimeError("模型未训练")

        scores = self.score_samples(X)
        thresh = threshold if threshold is not None else self._threshold
        is_anomaly = scores > thresh

        return is_anomaly, scores

    def save(self, path: str) -> None:
        """保存模型到文件

        Args:
            path: 文件路径
        """
        with open(path, "wb") as f:
            pickle.dump({
                "hidden_dims": self.hidden_dims,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "threshold_percentile": self.threshold_percentile,
                "is_fitted": self._is_fitted,
                "threshold": self._threshold,
                "use_pytorch": self._use_pytorch,
                "model_state": self._model.state_dict() if self._use_pytorch and self._model else None,
                "mean": self._mean,
                "cov_inv": self._cov_inv,
            }, f)

    def load(self, path: str) -> None:
        """从文件加载模型

        Args:
            path: 文件路径
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.hidden_dims = data["hidden_dims"]
        self.learning_rate = data["learning_rate"]
        self.epochs = data["epochs"]
        self.batch_size = data["batch_size"]
        self.threshold_percentile = data["threshold_percentile"]
        self._is_fitted = data["is_fitted"]
        self._threshold = data["threshold"]
        self._use_pytorch = data["use_pytorch"]
        self._mean = data["mean"]
        self._cov_inv = data["cov_inv"]

        if self._use_pytorch and data["model_state"]:
            import torch
            import torch.nn as nn
            
            # 重新创建模型结构
            class Autoencoder(nn.Module):
                def __init__(self, input_dim: int, hidden_dims: list[int]):
                    super().__init__()
                    encoder_layers = []
                    prev_dim = input_dim
                    for h_dim in hidden_dims:
                        encoder_layers.append(nn.Linear(prev_dim, h_dim))
                        encoder_layers.append(nn.ReLU())
                        prev_dim = h_dim

                    decoder_layers = []
                    for h_dim in reversed(hidden_dims[:-1]):
                        decoder_layers.append(nn.Linear(prev_dim, h_dim))
                        decoder_layers.append(nn.ReLU())
                        prev_dim = h_dim
                    decoder_layers.append(nn.Linear(prev_dim, input_dim))

                    self.encoder = nn.Sequential(*encoder_layers)
                    self.decoder = nn.Sequential(*decoder_layers)

                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded

            # 需要知道输入维度，这里假设为 hidden_dims[0] 的某个倍数
            input_dim = self.hidden_dims[0] * 2 if len(self.hidden_dims) > 0 else 10
            self._model = Autoencoder(input_dim, self.hidden_dims)
            self._model.load_state_dict(data["model_state"])
