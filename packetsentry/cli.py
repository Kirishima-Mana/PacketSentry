"""
PacketSentry 命令行界面

提供 analyze / train / monitor 子命令，
支持 PCAP 分析、模型训练和实时监控。
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from packetsentry import __version__
from packetsentry.utils.config import load_config
from packetsentry.utils.logger import logger
from packetsentry.collector.sniffer import PacketSniffer
from packetsentry.features.extractor import FeatureExtractor
from packetsentry.features.encoder import FeatureEncoder
from packetsentry.detector.engine import DetectionEngine
from packetsentry.visualizer.dashboard import Dashboard

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="PacketSentry")
def cli() -> None:
    """🛡️ PacketSentry — 基于机器学习的网络流量异常检测系统"""
    pass


@cli.command()
@click.argument("pcap_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(), help="配置文件路径")
@click.option("--model-dir", "-m", type=click.Path(), help="已训练模型目录")
@click.option("--output", "-o", type=click.Path(), help="输出报告路径（JSON）")
@click.option("--window-size", "-w", type=float, default=5.0, help="时间窗口大小（秒）")
def analyze(pcap_path: str, config: str, model_dir: str, output: str, window_size: float) -> None:
    """分析 PCAP 文件中的网络流量异常

    PCAP_PATH: 要分析的 PCAP 文件路径
    """
    cfg = load_config(config)
    if window_size != 5.0:
        cfg.features.window_size = window_size

    Dashboard.show_scan_header(pcap_path)

    # 读取 PCAP
    console.print("  [cyan]→[/cyan] 读取 PCAP 文件...")
    sniffer = PacketSniffer()
    packets = sniffer.read_pcap(pcap_path)

    if not packets:
        console.print("  [red]✗ 未读取到数据包[/red]")
        sys.exit(1)

    console.print(f"  [green]✓[/green] 读取 {len(packets)} 个数据包")

    # 创建检测引擎
    engine = DetectionEngine(cfg)

    if model_dir:
        # 加载已有模型
        engine.load_models(model_dir)
    else:
        # 使用当前数据训练（无监督，用自身数据作为基线）
        console.print("  [cyan]→[/cyan] 训练基线模型（无监督）...")
        engine.train(packets)

    # 执行检测
    console.print("  [cyan]→[/cyan] 执行异常检测...")
    report = engine.detect(packets)

    # 显示报告
    Dashboard.show_detection_report(report)

    # 输出 JSON
    if output:
        report_data = {
            "pcap_path": pcap_path,
            "total_windows": report.total_windows,
            "anomaly_windows": report.anomaly_windows,
            "anomaly_ratio": report.anomaly_ratio,
            "duration": report.duration,
            "anomaly_details": [
                {
                    "window_start": w.window_start,
                    "window_end": w.window_end,
                    "packet_count": w.packet_count,
                    "features": w.features,
                }
                for w in report.windows if w.is_anomaly
            ],
        }
        Path(output).write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"\n  [green]✓[/green] 报告已保存: {output}")


@cli.command()
@click.argument("pcap_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(), help="配置文件路径")
@click.option("--save-dir", "-s", type=click.Path(), default="models/trained", help="模型保存目录")
def train(pcap_path: str, config: str, save_dir: str) -> None:
    """使用正常流量 PCAP 训练检测模型

    PCAP_PATH: 正常流量 PCAP 文件路径
    """
    cfg = load_config(config)

    Dashboard.show_scan_header(f"训练模式: {pcap_path}")

    # 读取 PCAP
    console.print("  [cyan]→[/cyan] 读取正常流量 PCAP...")
    sniffer = PacketSniffer()
    packets = sniffer.read_pcap(pcap_path)

    if not packets:
        console.print("  [red]✗ 未读取到数据包[/red]")
        sys.exit(1)

    console.print(f"  [green]✓[/green] 读取 {len(packets)} 个数据包")

    # 训练模型
    engine = DetectionEngine(cfg)
    console.print("  [cyan]→[/cyan] 训练模型中...")
    engine.train(packets)

    # 保存模型
    engine.save_models(save_dir)
    console.print(f"  [green]✓[/green] 模型训练完成并保存到: {save_dir}")


@cli.command()
@click.option("--interface", "-i", default="eth0", help="网络接口")
@click.option("--config", "-c", type=click.Path(), help="配置文件路径")
@click.option("--model-dir", "-m", type=click.Path(), required=True, help="已训练模型目录")
@click.option("--timeout", "-t", type=int, default=60, help="监控时长（秒）")
def monitor(interface: str, config: str, model_dir: str, timeout: int) -> None:
    """实时监控网络流量（需要 root 权限）

    使用 --model-dir 加载已训练模型进行实时异常检测。
    """
    cfg = load_config(config)
    cfg.collector.interface = interface
    cfg.collector.timeout = timeout

    Dashboard.show_scan_header(f"实时监控: {interface}")

    # 加载模型
    engine = DetectionEngine(cfg)
    console.print(f"  [cyan]→[/cyan] 加载模型: {model_dir}")
    engine.load_models(model_dir)

    # 实时嗅探
    console.print(f"  [cyan]→[/cyan] 开始嗅探 ({timeout}s)...")
    sniffer = PacketSniffer(interface=interface, timeout=timeout)
    packets = sniffer.sniff_live(timeout=timeout)

    if not packets:
        console.print("  [yellow]⚠ 未捕获到数据包[/yellow]")
        sys.exit(0)

    console.print(f"  [green]✓[/green] 捕获 {len(packets)} 个数据包")

    # 检测
    report = engine.detect(packets)
    Dashboard.show_detection_report(report)


@cli.command()
def info() -> None:
    """显示系统信息和检测能力"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🛡️ PacketSentry[/bold cyan] — 系统信息\n"
            f"版本: {__version__}",
            border_style="cyan",
        )
    )

    # 检测依赖
    deps_table = click.Style("依赖检测", fg="cyan", bold=True)
    console.print(f"\n[bold cyan]📦 依赖检测[/bold cyan]")

    deps = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Scapy", "scapy"),
        ("Rich", "rich"),
        ("PyTorch", "torch"),
    ]

    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "已安装")
            console.print(f"  [green]✓[/green] {name}: {version}")
        except ImportError:
            console.print(f"  [red]✗[/red] {name}: 未安装")

    # 检测能力
    console.print(f"\n[bold cyan]🛡️ 检测能力[/bold cyan]")
    capabilities = [
        ("DDoS / DoS", "包速率突增、源 IP 聚集"),
        ("端口扫描", "目标端口分散度、SYN 比例异常"),
        ("数据渗出", "上行流量突增、大包比例异常"),
        ("横向移动", "内部主机间新连接激增"),
        ("协议异常", "非标准协议比例、端口-协议不匹配"),
    ]

    for name, desc in capabilities:
        console.print(f"  • {name}: [dim]{desc}[/dim]")


def main() -> None:
    """CLI 主入口"""
    cli()


if __name__ == "__main__":
    main()
