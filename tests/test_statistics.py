"""
PacketSentry 单元测试 - 特征统计计算
"""

import pytest
import numpy as np
from packetsentry.collector.parser import ParsedPacket
from packetsentry.features.statistics import (
    StatisticsCalculator,
    safe_mean,
    safe_std,
    safe_max,
    safe_min,
    entropy,
)


class TestStatisticsHelpers:
    """统计辅助函数测试"""

    def test_safe_mean(self):
        """测试安全均值计算"""
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert safe_mean([5.0]) == 5.0
        assert safe_mean([]) == 0.0

    def test_safe_std(self):
        """测试安全标准差计算"""
        assert safe_std([1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert safe_std([5.0]) == 0.0
        assert safe_std([]) == 0.0

    def test_safe_max(self):
        """测试安全最大值"""
        assert safe_max([1.0, 5.0, 3.0]) == 5.0
        assert safe_max([-1.0, -5.0]) == -1.0
        assert safe_max([]) == 0.0

    def test_safe_min(self):
        """测试安全最小值"""
        assert safe_min([1.0, 5.0, 3.0]) == 1.0
        assert safe_min([-1.0, -5.0]) == -5.0
        assert safe_min([]) == 0.0

    def test_entropy(self):
        """测试信息熵计算"""
        # 均匀分布熵最大
        assert entropy([1.0, 1.0, 1.0]) == pytest.approx(1.58496, rel=1e-5)
        # 单一分布熵为0
        assert entropy([1.0, 0.0, 0.0]) == 0.0
        # 空列表
        assert entropy([]) == 0.0
        assert entropy([0.0, 0.0]) == 0.0


class TestStatisticsCalculator:
    """统计特征计算器测试"""

    def _create_test_packets(self) -> list[ParsedPacket]:
        """创建测试数据包"""
        return [
            ParsedPacket(
                timestamp=1000.0,
                src_ip="192.168.1.100",
                dst_ip="192.168.1.1",
                src_port=54321,
                dst_port=80,
                protocol="TCP",
                packet_size=1500,
                payload_size=1460,
                tcp_flags="SYN",
            ),
            ParsedPacket(
                timestamp=1000.1,
                src_ip="192.168.1.100",
                dst_ip="192.168.1.1",
                src_port=54321,
                dst_port=80,
                protocol="TCP",
                packet_size=1000,
                payload_size=960,
                tcp_flags="ACK",
            ),
            ParsedPacket(
                timestamp=1000.2,
                src_ip="192.168.1.101",
                dst_ip="192.168.1.1",
                src_port=54322,
                dst_port=53,
                protocol="UDP",
                packet_size=500,
                payload_size=460,
            ),
        ]

    def test_compute_inter_arrival_stats(self):
        """测试包到达间隔统计"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_inter_arrival_stats(packets)
        assert "iat_mean" in stats
        assert "iat_std" in stats
        assert "iat_max" in stats
        assert "iat_min" in stats
        assert stats["iat_mean"] == pytest.approx(0.1, rel=1e-5)

    def test_compute_packet_size_stats(self):
        """测试包大小统计"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_packet_size_stats(packets)
        assert "pkt_size_mean" in stats
        assert "pkt_size_std" in stats
        assert "pkt_size_max" in stats
        assert "pkt_size_min" in stats
        assert stats["pkt_size_mean"] == pytest.approx(1000.0, rel=1e-5)
        assert stats["pkt_size_max"] == 1500.0
        assert stats["pkt_size_min"] == 500.0

    def test_compute_protocol_distribution(self):
        """测试协议分布"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_protocol_distribution(packets)
        assert stats["ratio_tcp"] == pytest.approx(2/3, rel=1e-5)
        assert stats["ratio_udp"] == pytest.approx(1/3, rel=1e-5)
        assert stats["ratio_icmp"] == 0.0
        assert stats["ratio_other"] == 0.0

    def test_compute_tcp_flags_distribution(self):
        """测试 TCP 标志位分布"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_tcp_flags_distribution(packets)
        assert stats["ratio_syn"] == pytest.approx(0.5, rel=1e-5)  # 2个TCP包，1个SYN
        assert stats["ratio_ack"] == pytest.approx(0.5, rel=1e-5)  # 2个TCP包，1个ACK
        assert stats["ratio_fin"] == 0.0
        assert stats["ratio_rst"] == 0.0
        assert stats["ratio_psh"] == 0.0

    def test_compute_ip_diversity(self):
        """测试 IP 多样性"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_ip_diversity(packets)
        assert stats["unique_src_ip"] == 2
        assert stats["unique_dst_ip"] == 1
        assert stats["unique_dst_port"] == 2  # 80 和 53
        assert stats["packets_per_src_ip"] == pytest.approx(1.5, rel=1e-5)

    def test_compute_traffic_direction_stats(self):
        """测试流量方向统计"""
        packets = self._create_test_packets()
        stats = StatisticsCalculator.compute_traffic_direction_stats(packets)
        assert stats["total_bytes"] == 3000.0
        assert stats["total_packets"] == 3.0
        assert stats["avg_pkt_size"] == 1000.0
        assert stats["payload_bytes"] == 2880.0
        assert stats["ratio_with_payload"] == 1.0
        assert stats["ratio_broadcast"] == 0.0
