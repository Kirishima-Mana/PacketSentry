"""
PacketSentry 终端可视化仪表盘

基于 Rich 库的实时流量监控和异常检测结果可视化。
"""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from packetsentry.detector.engine import DetectionReport
from packetsentry.features.extractor import FeatureWindow


console = Console()


class Dashboard:
    """终端仪表盘

    提供实时流量监控和异常检测结果的可视化展示，
    包括流量统计、协议分布、异常告警等。
    """

    @staticmethod
    def show_scan_header(target: str) -> None:
        """显示扫描头部信息

        Args:
            target: 扫描目标描述
        """
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]🛡️ PacketSentry[/bold cyan] — 网络流量异常检测\n"
                f"[dim]目标: {target}[/dim]",
                border_style="cyan",
            )
        )

    @staticmethod
    def show_training_progress(n_packets: int, n_features: int) -> None:
        """显示训练进度

        Args:
            n_packets: 数据包数量
            n_features: 特征维度
        """
        console.print(
            f"  [cyan]→[/cyan] 提取特征: {n_packets} 包 → {n_features} 特征向量"
        )

    @staticmethod
    def show_detection_report(report: DetectionReport) -> None:
        """显示检测结果报告

        Args:
            report: 检测报告
        """
        console.print()

        if report.total_windows == 0:
            console.print("  [yellow]⚠ 未检测到有效流量数据[/yellow]")
            return

        # 概览面板
        overview = Table(title="📊 检测概览", show_header=False, border_style="dim")
        overview.add_column("指标", style="bold")
        overview.add_column("值", justify="right")

        overview.add_row("时间窗口总数", str(report.total_windows))
        overview.add_row("异常窗口数", str(report.anomaly_windows))
        overview.add_row("异常比例", f"{report.anomaly_ratio * 100:.1f}%")
        overview.add_row("检测耗时", f"{report.duration:.2f}s")

        # 异常等级判定
        if report.anomaly_ratio > 0.5:
            level = "[bold red]🔴 高度异常[/bold red]"
        elif report.anomaly_ratio > 0.2:
            level = "[bold yellow]🟡 中度异常[/bold yellow]"
        elif report.anomaly_ratio > 0.05:
            level = "[bold blue]🔵 轻微异常[/bold blue]"
        else:
            level = "[bold green]✅ 流量正常[/bold green]"

        overview.add_row("威胁等级", level)
        console.print(overview)

        # 异常窗口详情
        anomaly_windows = [w for w in report.windows if w.is_anomaly]
        if anomaly_windows:
            console.print()
            anomaly_table = Table(
                title="⚠️ 异常时间窗口", show_lines=True, border_style="red"
            )
            anomaly_table.add_column("序号", width=6)
            anomaly_table.add_column("时间范围", width=30)
            anomaly_table.add_column("包数量", width=8)
            anomaly_table.add_column("特征摘要", width=40)

            for i, w in enumerate(anomaly_windows[:20], 1):
                # 生成特征摘要
                summary = Dashboard._feature_summary(w)
                anomaly_table.add_row(
                    str(i),
                    f"{w.window_start:.2f} - {w.window_end:.2f}",
                    str(w.packet_count),
                    summary,
                )

            console.print(anomaly_table)

            if len(anomaly_windows) > 20:
                console.print(
                    f"  [dim]... 还有 {len(anomaly_windows) - 20} 个异常窗口[/dim]"
                )

        # 模型对比（如果有集成结果）
        if report.ensemble_result is not None:
            console.print()
            model_table = Table(title="🤖 模型对比", border_style="dim")
            model_table.add_column("模型", width=20)
            model_table.add_column("检测异常数", width=12)
            model_table.add_column("异常比例", width=10)

            result = report.ensemble_result
            if_count = int(np_sum(result.if_predictions == -1))
            ae_count = int(np_sum(result.ae_predictions == -1))
            total = len(result.if_predictions)

            model_table.add_row(
                "Isolation Forest",
                str(if_count),
                f"{if_count/max(total,1)*100:.1f}%",
            )
            model_table.add_row(
                "Autoencoder",
                str(ae_count),
                f"{ae_count/max(total,1)*100:.1f}%",
            )
            model_table.add_row(
                "[bold]集成结果[/bold]",
                f"[bold]{report.anomaly_windows}[/bold]",
                f"[bold]{report.anomaly_ratio*100:.1f}%[/bold]",
            )

            console.print(model_table)

    @staticmethod
    def show_feature_statistics(windows: list[FeatureWindow]) -> None:
        """显示特征统计信息

        Args:
            windows: 特征窗口列表
        """
        if not windows:
            return

        console.print()
        table = Table(title="📈 特征统计", border_style="dim")
        table.add_column("特征", width=25)
        table.add_column("均值", width=12)
        table.add_column("最小值", width=12)
        table.add_column("最大值", width=12)

        # 选择关键特征展示
        key_features = [
            "total_packets", "total_bytes", "ratio_tcp",
            "ratio_syn", "unique_src_ip", "unique_dst_port",
            "iat_mean", "pkt_size_mean",
        ]

        for feat_name in key_features:
            values = [w.features.get(feat_name, 0.0) for w in windows]
            if values:
                import numpy as np
                table.add_row(
                    feat_name,
                    f"{np.mean(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                )

        console.print(table)

    @staticmethod
    def _feature_summary(window: FeatureWindow) -> str:
        """生成窗口特征摘要

        Args:
            window: 特征窗口

        Returns:
            摘要字符串
        """
        parts = []
        f = window.features

        if f.get("total_packets", 0) > 100:
            parts.append(f"高包率({f['total_packets']:.0f}/窗)")

        if f.get("ratio_syn", 0) > 0.5:
            parts.append(f"SYN异常({f['ratio_syn']:.0%})")

        if f.get("unique_dst_port", 0) > 50:
            parts.append(f"端口扫描({f['unique_dst_port']:.0f}端口)")

        if f.get("unique_src_ip", 0) > 20:
            parts.append(f"多源IP({f['unique_src_ip']:.0f})")

        if f.get("ratio_icmp", 0) > 0.3:
            parts.append(f"ICMP异常({f['ratio_icmp']:.0%})")

        if not parts:
            parts.append("综合异常")

        return "; ".join(parts)


def np_sum(arr) -> int:
    """安全计算数组求和"""
    import numpy as np
    return int(np.sum(arr))
