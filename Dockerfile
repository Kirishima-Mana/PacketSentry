# ============================================================
# PacketSentry Dockerfile — 多阶段构建
# 基于机器学习的网络流量异常检测系统
# ============================================================

# ---------- 阶段1：构建依赖 ----------
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装系统依赖（Scapy 需要 libpcap）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- 阶段2：运行时镜像 ----------
FROM python:3.12-slim AS runtime

LABEL maintainer="PacketSentry Contributors"
LABEL description="PacketSentry - 基于机器学习的网络流量异常检测系统"
LABEL version="0.1.0"

# 安装运行时系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpcap-dev \
        tcpdump \
        iproute2 \
        net-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制 Python 依赖
COPY --from=builder /install /usr/local

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装 PacketSentry 包
RUN pip install --no-cache-dir -e .

# 创建非 root 用户运行
RUN groupadd -r packetsentry && \
    useradd -r -g packetsentry -m packetsentry && \
    mkdir -p /data && \
    chown -R packetsentry:packetsentry /app /data

USER packetsentry

# 数据目录挂载点
VOLUME ["/data"]

# 默认环境变量
ENV PYTHONUNBUFFERED=1
ENV PACKETSENTRY_CONFIG=/app/config/default.ini

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from packetsentry.utils.logger import logger; print('OK')" || exit 1

# 默认入口：显示帮助
ENTRYPOINT ["python", "-m", "packetsentry"]
CMD ["--help"]
