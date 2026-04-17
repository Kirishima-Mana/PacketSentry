"""
PacketSentry 单元测试 - 数据包解析器
"""

import pytest
from packetsentry.collector.parser import PacketParser, ParsedPacket


class TestParsedPacket:
    """ParsedPacket 数据类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        packet = ParsedPacket(
            timestamp=1234567890.123,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.1",
            src_port=54321,
            dst_port=80,
            protocol="TCP",
            packet_size=1500,
            payload_size=1460,
        )
        assert packet.timestamp == 1234567890.123
        assert packet.src_ip == "192.168.1.100"
        assert packet.dst_ip == "192.168.1.1"
        assert packet.src_port == 54321
        assert packet.dst_port == 80
        assert packet.protocol == "TCP"
        assert packet.packet_size == 1500
        assert packet.payload_size == 1460

    def test_to_dict(self):
        """测试转换为字典"""
        packet = ParsedPacket(
            timestamp=1234567890.123,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.1",
            src_port=54321,
            dst_port=80,
            protocol="TCP",
            packet_size=1500,
            payload_size=1460,
            tcp_flags="SYN,ACK",
        )
        data = packet.to_dict()
        assert data["timestamp"] == 1234567890.123
        assert data["src_ip"] == "192.168.1.100"
        assert data["dst_ip"] == "192.168.1.1"
        assert data["src_port"] == 54321
        assert data["dst_port"] == 80
        assert data["protocol"] == "TCP"
        assert data["packet_size"] == 1500
        assert data["payload_size"] == 1460
        assert data["tcp_flags"] == "SYN,ACK"


class TestPacketParser:
    """数据包解析器测试"""

    def test_parse_from_dict(self):
        """测试从字典解析"""
        parser = PacketParser()
        packet_dict = {
            "timestamp": 1234567890.123,
            "src_ip": "192.168.1.100",
            "dst_ip": "192.168.1.1",
            "src_port": 54321,
            "dst_port": 80,
            "protocol": "TCP",
            "packet_size": 1500,
            "payload_size": 1460,
            "ttl": 64,
            "tcp_flags": "SYN,ACK",
            "tcp_window_size": 8192,
            "is_broadcast": False,
            "layer7_protocol": "HTTP",
        }
        packet = parser.parse_from_dict(packet_dict)
        assert packet.timestamp == 1234567890.123
        assert packet.src_ip == "192.168.1.100"
        assert packet.dst_ip == "192.168.1.1"
        assert packet.src_port == 54321
        assert packet.dst_port == 80
        assert packet.protocol == "TCP"
        assert packet.packet_size == 1500
        assert packet.payload_size == 1460
        assert packet.ttl == 64
        assert packet.tcp_flags == "SYN,ACK"
        assert packet.tcp_window_size == 8192
        assert packet.is_broadcast is False
        assert packet.layer7_protocol == "HTTP"

    def test_decode_tcp_flags(self):
        """测试 TCP 标志位解码"""
        parser = PacketParser()
        assert parser._decode_tcp_flags("S") == "SYN"
        assert parser._decode_tcp_flags("SA") == "SYN,ACK"
        assert parser._decode_tcp_flags("PA") == "PSH,ACK"
        assert parser._decode_tcp_flags("FA") == "FIN,ACK"
        assert parser._decode_tcp_flags("RA") == "RST,ACK"
        assert parser._decode_tcp_flags("") == ""
        assert parser._decode_tcp_flags("X") == "X"  # 未知标志

    def test_infer_l7_protocol(self):
        """测试应用层协议推断"""
        parser = PacketParser()
        assert parser._infer_l7_protocol(80, 54321, "TCP") == "HTTP"
        assert parser._infer_l7_protocol(443, 54321, "TCP") == "HTTPS"
        assert parser._infer_l7_protocol(53, 54321, "UDP") == "DNS"
        assert parser._infer_l7_protocol(22, 54321, "TCP") == "SSH"
        assert parser._infer_l7_protocol(8080, 54321, "TCP") == "HTTP-PROXY"
        assert parser._infer_l7_protocol(9999, 54321, "TCP") == "HIGH-PORT"
        assert parser._infer_l7_protocol(443, 54321, "UDP") == "UDP-443"
