"""
PacketSentry 协议解析模块

基于 Scapy 解析原始数据包，提取协议头部信息和有效载荷特征。
支持 Ethernet / IP / TCP / UDP / ICMP / DNS / HTTP 等协议。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ParsedPacket:
    """解析后的数据包信息

    封装从原始数据包中提取的所有协议信息，
    作为后续特征提取的输入。
    """
    timestamp: float                     # 时间戳
    src_ip: str = ""                     # 源 IP 地址
    dst_ip: str = ""                     # 目标 IP 地址
    src_port: int = 0                    # 源端口号
    dst_port: int = 0                    # 目标端口号
    protocol: str = "OTHER"              # 传输层协议（TCP/UDP/ICMP/OTHER）
    ip_version: int = 4                  # IP 版本（4/6）
    packet_size: int = 0                 # 数据包总大小（字节）
    payload_size: int = 0                # 有效载荷大小（字节）
    ttl: int = 0                         # TTL 值
    # TCP 特有字段
    tcp_flags: str = ""                  # TCP 标志位（如 SYN,ACK）
    tcp_window_size: int = 0             # TCP 窗口大小
    # 统计标记
    is_broadcast: bool = False           # 是否广播/组播
    layer7_protocol: str = ""            # 应用层协议（HTTP/DNS/SSL 等）

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "packet_size": self.packet_size,
            "payload_size": self.payload_size,
            "tcp_flags": self.tcp_flags,
        }


class PacketParser:
    """数据包协议解析器

    将 Scapy 捕获的原始数据包转换为结构化的 ParsedPacket 对象，
    提取各层协议的关键信息用于安全分析。
    """

    # TCP 标志位名称映射
    TCP_FLAG_MAP = {
        "F": "FIN", "S": "SYN", "R": "RST", "P": "PSH",
        "A": "ACK", "U": "URG", "E": "ECE", "C": "CWR",
    }

    # 常见端口与协议映射
    WELL_KNOWN_PORTS: dict[int, str] = {
        20: "FTP-DATA", 21: "FTP", 22: "SSH", 23: "TELNET",
        25: "SMTP", 53: "DNS", 67: "DHCP", 68: "DHCP",
        80: "HTTP", 110: "POP3", 123: "NTP", 143: "IMAP",
        161: "SNMP", 162: "SNMP-TRAP", 443: "HTTPS", 445: "SMB",
        993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 1521: "ORACLE",
        3306: "MYSQL", 3389: "RDP", 5432: "POSTGRESQL",
        5900: "VNC", 6379: "REDIS", 8080: "HTTP-PROXY",
        8443: "HTTPS-ALT", 27017: "MONGODB",
    }

    def parse(self, raw_packet) -> Optional[ParsedPacket]:
        """解析单个原始数据包

        Args:
            raw_packet: Scapy 捕获的原始数据包对象

        Returns:
            ParsedPacket 或 None（非 IP 包时返回 None）
        """
        try:
            return self._parse_internal(raw_packet)
        except Exception:
            return None

    def _parse_internal(self, pkt) -> Optional[ParsedPacket]:
        """内部解析逻辑

        逐层解析数据包协议栈，提取关键字段。

        Args:
            pkt: Scapy 数据包

        Returns:
            ParsedPacket 对象
        """
        # 尝试解析 IP 层
        if not hasattr(pkt, "ip") and not hasattr(pkt, "IPv6"):
            # 尝试 Scapy 的 IP / IPv6 属性
            ip_layer = None
            if pkt.haslayer("IP"):
                ip_layer = pkt["IP"]
            elif pkt.haslayer("IPv6"):
                ip_layer = pkt["IPv6"]
            else:
                return None  # 非 IP 数据包
        else:
            ip_layer = pkt.getlayer("IP") or pkt.getlayer("IPv6")

        if ip_layer is None:
            return None

        parsed = ParsedPacket(timestamp=float(pkt.time))

        # IP 层信息
        parsed.src_ip = getattr(ip_layer, "src", "")
        parsed.dst_ip = getattr(ip_layer, "dst", "")
        parsed.ttl = getattr(ip_layer, "ttl", 0)
        parsed.ip_version = getattr(ip_layer, "version", 4)
        parsed.packet_size = len(pkt)

        # 广播/组播检测
        if parsed.dst_ip:
            parsed.is_broadcast = (
                parsed.dst_ip.endswith(".255")  # 广播
                or parsed.dst_ip.startswith("224.")  # 组播
                or parsed.dst_ip.startswith("239.")  # 组播
            )

        # 传输层解析
        if pkt.haslayer("TCP"):
            tcp = pkt["TCP"]
            parsed.protocol = "TCP"
            parsed.src_port = tcp.sport
            parsed.dst_port = tcp.dport
            parsed.tcp_flags = self._decode_tcp_flags(str(tcp.flags))
            parsed.tcp_window_size = tcp.window
            parsed.payload_size = len(tcp.payload) if tcp.payload else 0
            parsed.layer7_protocol = self._infer_l7_protocol(
                parsed.dst_port, parsed.src_port, "TCP"
            )

        elif pkt.haslayer("UDP"):
            udp = pkt["UDP"]
            parsed.protocol = "UDP"
            parsed.src_port = udp.sport
            parsed.dst_port = udp.dport
            parsed.payload_size = len(udp.payload) if udp.payload else 0
            parsed.layer7_protocol = self._infer_l7_protocol(
                parsed.dst_port, parsed.src_port, "UDP"
            )

        elif pkt.haslayer("ICMP"):
            parsed.protocol = "ICMP"
            parsed.payload_size = len(pkt["ICMP"].payload) if pkt["ICMP"].payload else 0

        else:
            parsed.protocol = "OTHER"
            parsed.payload_size = parsed.packet_size - len(ip_layer)

        return parsed

    def _decode_tcp_flags(self, flags_str: str) -> str:
        """解码 TCP 标志位字符串

        将 Scapy 的标志位表示（如 'S', 'SA', 'PA'）
        转换为可读格式（如 'SYN', 'SYN,ACK'）。

        Args:
            flags_str: Scapy 格式的 TCP 标志位

        Returns:
            可读的标志位字符串
        """
        result = []
        for char in flags_str:
            if char in self.TCP_FLAG_MAP:
                result.append(self.TCP_FLAG_MAP[char])
        return ",".join(result) if result else flags_str

    def _infer_l7_protocol(self, dst_port: int, src_port: int, transport: str) -> str:
        """根据端口号推断应用层协议

        Args:
            dst_port: 目标端口
            src_port: 源端口
            transport: 传输层协议

        Returns:
            推断的应用层协议名称
        """
        # 优先检查目标端口
        if dst_port in self.WELL_KNOWN_PORTS:
            proto = self.WELL_KNOWN_PORTS[dst_port]
            # HTTPS 在 TCP 443
            if proto == "HTTPS" and transport != "TCP":
                return "UDP-443"
            return proto

        # 再检查源端口
        if src_port in self.WELL_KNOWN_PORTS:
            return self.WELL_KNOWN_PORTS[src_port]

        # 高端口通常是客户端
        if dst_port > 1024:
            return "HIGH-PORT"
        return "UNKNOWN"

    def parse_from_dict(self, pkt_dict: dict) -> ParsedPacket:
        """从字典构建 ParsedPacket（用于离线/测试数据）

        Args:
            pkt_dict: 数据包字典

        Returns:
            ParsedPacket 对象
        """
        return ParsedPacket(
            timestamp=pkt_dict.get("timestamp", 0.0),
            src_ip=pkt_dict.get("src_ip", ""),
            dst_ip=pkt_dict.get("dst_ip", ""),
            src_port=pkt_dict.get("src_port", 0),
            dst_port=pkt_dict.get("dst_port", 0),
            protocol=pkt_dict.get("protocol", "OTHER"),
            packet_size=pkt_dict.get("packet_size", 0),
            payload_size=pkt_dict.get("payload_size", 0),
            ttl=pkt_dict.get("ttl", 0),
            tcp_flags=pkt_dict.get("tcp_flags", ""),
            tcp_window_size=pkt_dict.get("tcp_window_size", 0),
            is_broadcast=pkt_dict.get("is_broadcast", False),
            layer7_protocol=pkt_dict.get("layer7_protocol", ""),
        )
