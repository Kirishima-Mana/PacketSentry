"""
PacketSentry 网络流重组模块

将离散数据包按五元组（src_ip, dst_ip, src_port, dst_port, protocol）
重组为双向网络流，计算流级统计特征。
"""

from dataclasses import dataclass, field
from typing import Optional

from packetsentry.collector.parser import ParsedPacket


@dataclass
class NetworkFlow:
    """双向网络流

    基于五元组将数据包聚合为流，记录流的生命周期统计信息。
    """
    # 五元组标识
    src_ip: str                          # 源 IP
    dst_ip: str                          # 目标 IP
    src_port: int                        # 源端口
    dst_port: int                        # 目标端口
    protocol: str                        # 传输层协议

    # 流统计
    start_time: float = 0.0             # 流开始时间
    end_time: float = 0.0               # 流结束时间
    packets_forward: int = 0             # 正向包数量（src→dst）
    packets_backward: int = 0            # 反向包数量（dst→src）
    bytes_forward: int = 0               # 正向字节总数
    bytes_backward: int = 0              # 反向字节总数
    tcp_flags_forward: list[str] = field(default_factory=list)
    tcp_flags_backward: list[str] = field(default_factory=list)

    # 推断信息
    layer7_protocol: str = ""            # 应用层协议

    @property
    def flow_id(self) -> str:
        """流唯一标识（五元组）"""
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}-{self.protocol}"

    @property
    def duration(self) -> float:
        """流持续时间（秒）"""
        return max(0.0, self.end_time - self.start_time)

    @property
    def total_packets(self) -> int:
        """总包数"""
        return self.packets_forward + self.packets_backward

    @property
    def total_bytes(self) -> int:
        """总字节数"""
        return self.bytes_forward + self.bytes_backward

    @property
    def is_bidirectional(self) -> bool:
        """是否为双向流"""
        return self.packets_forward > 0 and self.packets_backward > 0


class FlowReassembler:
    """网络流重组器

    将数据包按五元组聚合成流，维护活跃流表，
    支持超时淘汰和流完成回调。
    """

    def __init__(self, flow_timeout: float = 120.0) -> None:
        """初始化流重组器

        Args:
            flow_timeout: 流超时时间（秒），超过此时间无新包则认为流结束
        """
        self.flow_timeout = flow_timeout
        self._active_flows: dict[str, NetworkFlow] = {}
        self._completed_flows: list[NetworkFlow] = []

    def add_packet(self, packet: ParsedPacket) -> Optional[NetworkFlow]:
        """添加数据包到流表

        根据五元组查找或创建对应的流，更新流统计信息。
        当流完成或超时时返回该流。

        Args:
            packet: 解析后的数据包

        Returns:
            若流完成则返回该流，否则返回 None
        """
        # 构建正向和反向五元组
        forward_key = (
            f"{packet.src_ip}:{packet.src_port}-"
            f"{packet.dst_ip}:{packet.dst_port}-{packet.protocol}"
        )
        reverse_key = (
            f"{packet.dst_ip}:{packet.dst_port}-"
            f"{packet.src_ip}:{packet.src_port}-{packet.protocol}"
        )

        # 查找已有流
        flow = self._active_flows.get(forward_key)
        is_forward = True

        if flow is None:
            flow = self._active_flows.get(reverse_key)
            is_forward = False

        if flow is None:
            # 创建新流
            flow = NetworkFlow(
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                protocol=packet.protocol,
                start_time=packet.timestamp,
                end_time=packet.timestamp,
                layer7_protocol=packet.layer7_protocol,
            )
            self._active_flows[forward_key] = flow

            # 正向第一个包
            flow.packets_forward = 1
            flow.bytes_forward = packet.packet_size
            if packet.tcp_flags:
                flow.tcp_flags_forward.append(packet.tcp_flags)
            return None

        # 更新已有流
        flow.end_time = packet.timestamp

        if is_forward:
            flow.packets_forward += 1
            flow.bytes_forward += packet.packet_size
            if packet.tcp_flags:
                flow.tcp_flags_forward.append(packet.tcp_flags)
        else:
            flow.packets_backward += 1
            flow.bytes_backward += packet.packet_size
            if packet.tcp_flags:
                flow.tcp_flags_backward.append(packet.tcp_flags)

        # 检查流是否完成（TCP FIN/RST）
        if packet.protocol == "TCP" and packet.tcp_flags:
            if "FIN" in packet.tcp_flags or "RST" in packet.tcp_flags:
                completed_flow = self._active_flows.pop(
                    forward_key if is_forward else reverse_key, None
                )
                if completed_flow:
                    self._completed_flows.append(completed_flow)
                    return completed_flow

        return None

    def check_timeouts(self, current_time: float) -> list[NetworkFlow]:
        """检查超时流并回收

        Args:
            current_time: 当前时间戳

        Returns:
            超时完成的流列表
        """
        timed_out: list[NetworkFlow] = []
        expired_keys: list[str] = []

        for key, flow in self._active_flows.items():
            if current_time - flow.end_time > self.flow_timeout:
                timed_out.append(flow)
                expired_keys.append(key)

        for key in expired_keys:
            del self._active_flows[key]

        self._completed_flows.extend(timed_out)
        return timed_out

    def get_active_flow_count(self) -> int:
        """获取活跃流数量"""
        return len(self._active_flows)

    def get_completed_flows(self) -> list[NetworkFlow]:
        """获取并清空已完成流列表"""
        flows = self._completed_flows.copy()
        self._completed_flows.clear()
        return flows

    def flush_all(self) -> list[NetworkFlow]:
        """强制完成所有活跃流

        Returns:
            所有流（活跃 + 已完成）
        """
        all_flows = list(self._active_flows.values()) + self._completed_flows
        self._active_flows.clear()
        self._completed_flows.clear()
        return all_flows
