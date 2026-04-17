"""
PacketSentry 数据包嗅探引擎

基于 Scapy 的数据包捕获模块，支持实时网卡嗅探和 PCAP 文件读取。
"""

import time
from pathlib import Path
from typing import Optional, Callable

from packetsentry.collector.parser import PacketParser, ParsedPacket
from packetsentry.collector.flow import FlowReassembler, NetworkFlow
from packetsentry.utils.logger import logger


class PacketSniffer:
    """数据包嗅探引擎

    支持两种工作模式：
    1. 实时嗅探：从指定网卡捕获实时流量
    2. 离线分析：读取 PCAP 文件进行回放分析
    """

    def __init__(
        self,
        interface: str = "eth0",
        bpf_filter: str = "",
        flow_timeout: float = 120.0,
        packet_callback: Optional[Callable[[ParsedPacket], None]] = None,
    ) -> None:
        """初始化嗅探引擎

        Args:
            interface: 网卡接口名
            bpf_filter: BPF 过滤器表达式
            flow_timeout: 流超时时间
            packet_callback: 数据包回调函数
        """
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.parser = PacketParser()
        self.flow_reassembler = FlowReassembler(flow_timeout=flow_timeout)
        self.packet_callback = packet_callback

        # 统计信息
        self.packets_captured: int = 0
        self.packets_parsed: int = 0
        self.packets_dropped: int = 0
        self.flows_completed: int = 0

    def sniff_live(
        self,
        timeout: int = 60,
        max_packets: int = 0,
    ) -> list[ParsedPacket]:
        """实时嗅探网络流量

        Args:
            timeout: 嗅探超时（秒）
            max_packets: 最大捕获包数（0=无限）

        Returns:
            捕获并解析的数据包列表
        """
        try:
            from scapy.all import sniff
        except ImportError:
            logger.error("Scapy 未安装，请运行: pip install scapy")
            return []

        logger.info(f"开始实时嗅探 [interface={self.interface}, timeout={timeout}s]")

        packets: list[ParsedPacket] = []

        def _process_packet(raw_pkt):
            parsed = self.parser.parse(raw_pkt)
            if parsed:
                packets.append(parsed)
                self.packets_parsed += 1
                self.flow_reassembler.add_packet(parsed)

                if self.packet_callback:
                    self.packet_callback(parsed)
            else:
                self.packets_dropped += 1

            self.packets_captured += 1

        try:
            kwargs = {
                "iface": self.interface,
                "prn": _process_packet,
                "timeout": timeout,
                "store": False,
            }
            if self.bpf_filter:
                kwargs["filter"] = self.bpf_filter
            if max_packets > 0:
                kwargs["count"] = max_packets

            sniff(**kwargs)
        except PermissionError:
            logger.error("权限不足，实时嗅探需要 root 权限")
        except Exception as e:
            logger.error(f"嗅探异常: {e}")

        logger.info(
            f"嗅探完成: 捕获 {self.packets_captured} 包, "
            f"解析 {self.packets_parsed}, 丢弃 {self.packets_dropped}"
        )
        return packets

    def read_pcap(self, pcap_path: str) -> list[ParsedPacket]:
        """读取 PCAP 文件进行离线分析

        Args:
            pcap_path: PCAP 文件路径

        Returns:
            解析后的数据包列表
        """
        try:
            from scapy.all import rdpcap
        except ImportError:
            logger.error("Scapy 未安装，请运行: pip install scapy")
            return []

        path = Path(pcap_path)
        if not path.exists():
            logger.error(f"PCAP 文件不存在: {pcap_path}")
            return []

        logger.info(f"读取 PCAP 文件: {pcap_path}")

        try:
            raw_packets = rdpcap(str(path))
        except Exception as e:
            logger.error(f"PCAP 文件读取失败: {e}")
            return []

        packets: list[ParsedPacket] = []

        for raw_pkt in raw_packets:
            parsed = self.parser.parse(raw_pkt)
            if parsed:
                packets.append(parsed)
                self.packets_parsed += 1
                self.flow_reassembler.add_packet(parsed)

                if self.packet_callback:
                    self.packet_callback(parsed)
            else:
                self.packets_dropped += 1

            self.packets_captured += 1

        # 强制完成所有活跃流
        final_flows = self.flow_reassembler.flush_all()
        self.flows_completed += len(final_flows)

        logger.info(
            f"PCAP 分析完成: {self.packets_captured} 包, "
            f"解析 {self.packets_parsed}, 丢弃 {self.packets_dropped}"
        )
        return packets

    def get_all_flows(self) -> list[NetworkFlow]:
        """获取所有网络流（活跃+已完成）"""
        return self.flow_reassembler.flush_all()
