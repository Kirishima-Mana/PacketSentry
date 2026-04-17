"""
PacketSentry 单元测试 - 特征提取器
"""

import pytest
from packetsentry.collector.parser import ParsedPacket
from packetsentry.features.extractor import FeatureExtractor, FeatureWindow


class TestFeatureExtractor:
    """特征提取器测试"""

    def _create_test_packets(self) -> list[ParsedPacket]:
        """创建测试数据包序列"""
        packets = []
        base_time = 1000.0
        for i in range(10):
            packet = ParsedPacket(
                timestamp=base_time + i * 0.5,  # 每 0.5 秒一个包
                src_ip=f"192.168.1.{i % 5 + 1}",
                dst_ip="192.168.1.1",
                src_port=50000 + i,
                dst_port=80 if i < 5 else 443,
                protocol="TCP" if i < 7 else "UDP",
                packet_size=1000 + i * 100,
                payload_size=960 + i * 100,
                tcp_flags="SYN" if i == 0 else "ACK",
            )
            packets.append(packet)
        return packets

    def test_extract_windows(self):
        """测试时间窗口提取"""
        extractor = FeatureExtractor(window_size=2.0, window_step=1.0)
        packets = self._create_test_packets()
        windows = extractor.extract(packets)

        assert len(windows) > 0
        for window in windows:
            assert isinstance(window, FeatureWindow)
            assert window.window_start < window.window_end
            assert window.packet_count >= 0
            assert isinstance(window.features, dict)
            assert len(window.features) > 0

    def test_window_size_step(self):
        """测试窗口大小和步长"""
        extractor = FeatureExtractor(window_size=3.0, window_step=1.5)
        packets = self._create_test_packets()
        windows = extractor.extract(packets)

        if len(windows) >= 2:
            # 检查窗口步长
            step = windows[1].window_start - windows[0].window_start
            assert step == pytest.approx(1.5, rel=1e-5)

            # 检查窗口大小
            for window in windows:
                duration = window.window_end - window.window_start
                assert duration == pytest.approx(3.0, rel=1e-5)

    def test_empty_packets(self):
        """测试空数据包列表"""
        extractor = FeatureExtractor()
        windows = extractor.extract([])
        assert len(windows) == 0

    def test_extract_feature_matrix(self):
        """测试特征矩阵提取"""
        extractor = FeatureExtractor(window_size=2.0, window_step=1.0)
        packets = self._create_test_packets()
        windows, feature_dicts = extractor.extract_feature_matrix(packets)

        assert len(windows) == len(feature_dicts)
        assert len(windows) > 0

        for i, window in enumerate(windows):
            assert window.features == feature_dicts[i]

    def test_feature_completeness(self):
        """测试特征完整性"""
        extractor = FeatureExtractor()
        packets = self._create_test_packets()
        windows = extractor.extract(packets)

        if windows:
            features = windows[0].features
            # 检查关键特征是否存在
            expected_features = [
                "total_packets", "total_bytes", "avg_pkt_size",
                "iat_mean", "iat_std", "pkt_size_mean", "pkt_size_std",
                "ratio_tcp", "ratio_udp", "ratio_syn", "ratio_ack",
                "unique_src_ip", "unique_dst_ip", "unique_dst_port",
            ]
            for feat in expected_features:
                assert feat in features
                assert isinstance(features[feat], (int, float))

    def test_window_packet_count(self):
        """测试窗口内包数量统计"""
        extractor = FeatureExtractor(window_size=2.0, window_step=1.0)
        packets = self._create_test_packets()
        windows = extractor.extract(packets)

        for window in windows:
            # 验证包数量与特征中的 total_packets 一致
            assert window.packet_count == int(window.features.get("total_packets", 0))
