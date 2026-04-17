"""
PacketSentry 统计特征计算模块

在网络流数据基础上计算各类统计特征，包括包速率、
字节分布、协议比例、TCP 标志位分布等。
"""

import numpy as np
from typing import Optional

from packetsentry.collector.parser import ParsedPacket


def safe_mean(values: list[float]) -> float:
    """安全计算均值，空列表返回 0.0"""
    return float(np.mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    """安全计算标准差，空列表或单元素返回 0.0"""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def safe_max(values: list[float]) -> float:
    """安全计算最大值"""
    return float(max(values)) if values else 0.0


def safe_min(values: list[float]) -> float:
    """安全计算最小值"""
    return float(min(values)) if values else 0.0


def entropy(values: list[float]) -> float:
    """计算信息熵

    用于衡量分布的均匀程度。分布越均匀，熵越大。

    Args:
        values: 数值列表

    Returns:
        信息熵值
    """
    if not values or sum(values) == 0:
        return 0.0

    total = sum(values)
    probs = [v / total for v in values if v > 0]
    return float(-sum(p * np.log2(p) for p in probs))


class StatisticsCalculator:
    """统计特征计算器

    对时间窗口内的数据包集合计算多维统计特征。
    """

    @staticmethod
    def compute_inter_arrival_stats(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算包到达时间间隔统计

        Args:
            packets: 时间窗口内的数据包列表

        Returns:
            包含时间间隔统计特征的字典
        """
        if len(packets) < 2:
            return {
                "iat_mean": 0.0,
                "iat_std": 0.0,
                "iat_max": 0.0,
                "iat_min": 0.0,
            }

        timestamps = sorted(p.timestamp for p in packets)
        intervals = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]

        return {
            "iat_mean": safe_mean(intervals),
            "iat_std": safe_std(intervals),
            "iat_max": safe_max(intervals),
            "iat_min": safe_min(intervals),
        }

    @staticmethod
    def compute_packet_size_stats(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算包大小统计

        Args:
            packets: 数据包列表

        Returns:
            包大小统计特征字典
        """
        if not packets:
            return {
                "pkt_size_mean": 0.0,
                "pkt_size_std": 0.0,
                "pkt_size_max": 0.0,
                "pkt_size_min": 0.0,
            }

        sizes = [float(p.packet_size) for p in packets]

        return {
            "pkt_size_mean": safe_mean(sizes),
            "pkt_size_std": safe_std(sizes),
            "pkt_size_max": safe_max(sizes),
            "pkt_size_min": safe_min(sizes),
        }

    @staticmethod
    def compute_protocol_distribution(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算协议分布比例

        Args:
            packets: 数据包列表

        Returns:
            协议比例特征字典
        """
        total = len(packets)
        if total == 0:
            return {"ratio_tcp": 0.0, "ratio_udp": 0.0, "ratio_icmp": 0.0, "ratio_other": 0.0}

        counts = {"TCP": 0, "UDP": 0, "ICMP": 0, "OTHER": 0}
        for p in packets:
            proto = p.protocol.upper()
            if proto in counts:
                counts[proto] += 1
            else:
                counts["OTHER"] += 1

        return {
            "ratio_tcp": counts["TCP"] / total,
            "ratio_udp": counts["UDP"] / total,
            "ratio_icmp": counts["ICMP"] / total,
            "ratio_other": counts["OTHER"] / total,
        }

    @staticmethod
    def compute_tcp_flags_distribution(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算 TCP 标志位分布

        Args:
            packets: 数据包列表

        Returns:
            TCP 标志位比例字典
        """
        tcp_packets = [p for p in packets if p.protocol == "TCP"]
        total_tcp = len(tcp_packets)

        if total_tcp == 0:
            return {
                "ratio_syn": 0.0, "ratio_ack": 0.0,
                "ratio_fin": 0.0, "ratio_rst": 0.0,
                "ratio_psh": 0.0,
            }

        flag_counts = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0}
        for p in tcp_packets:
            if p.tcp_flags:
                for flag in p.tcp_flags.split(","):
                    flag = flag.strip()
                    if flag in flag_counts:
                        flag_counts[flag] += 1

        return {
            "ratio_syn": flag_counts["SYN"] / total_tcp,
            "ratio_ack": flag_counts["ACK"] / total_tcp,
            "ratio_fin": flag_counts["FIN"] / total_tcp,
            "ratio_rst": flag_counts["RST"] / total_tcp,
            "ratio_psh": flag_counts["PSH"] / total_tcp,
        }

    @staticmethod
    def compute_ip_diversity(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算 IP 地址多样性特征

        高唯一 IP 数可能是 DDoS 或扫描的指标。

        Args:
            packets: 数据包列表

        Returns:
            IP 多样性特征字典
        """
        src_ips = set()
        dst_ips = set()
        dst_ports = set()

        for p in packets:
            if p.src_ip:
                src_ips.add(p.src_ip)
            if p.dst_ip:
                dst_ips.add(p.dst_ip)
            if p.dst_port:
                dst_ports.add(p.dst_port)

        total = len(packets) or 1

        return {
            "unique_src_ip": len(src_ips),
            "unique_dst_ip": len(dst_ips),
            "unique_dst_port": len(dst_ports),
            "src_ip_entropy": entropy([1.0] * len(src_ips)) if src_ips else 0.0,
            "dst_port_entropy": entropy([1.0] * len(dst_ports)) if dst_ports else 0.0,
            "packets_per_src_ip": total / max(len(src_ips), 1),
        }

    @staticmethod
    def compute_traffic_direction_stats(packets: list[ParsedPacket]) -> dict[str, float]:
        """计算流量方向统计

        区分上行/下行流量，数据渗出通常表现为异常的上行流量比例。

        Args:
            packets: 数据包列表

        Returns:
            流量方向特征字典
        """
        total_bytes = sum(p.packet_size for p in packets)
        total_packets = len(packets)
        payload_bytes = sum(p.payload_size for p in packets)

        # 包含有效载荷的包占比
        packets_with_payload = sum(1 for p in packets if p.payload_size > 0)
        ratio_with_payload = packets_with_payload / max(total_packets, 1)

        # 广播/组播包占比
        broadcast_count = sum(1 for p in packets if p.is_broadcast)
        ratio_broadcast = broadcast_count / max(total_packets, 1)

        return {
            "total_bytes": float(total_bytes),
            "total_packets": float(total_packets),
            "avg_pkt_size": float(total_bytes) / max(total_packets, 1),
            "payload_bytes": float(payload_bytes),
            "ratio_with_payload": ratio_with_payload,
            "ratio_broadcast": ratio_broadcast,
        }
