# hardware/ft_sensor_client.py
"""
DG-5F-M 개발자 모드용 핑거팁 F/T 센서 리더

- 통신 방식: TCP/IP
- 명령어: Get Data(0x01)
- 데이터 종류: F/T Sensor(0x05)
- 반환 순서(센서당): Fx, Fy, Fz, Tx, Ty, Tz
- 데이터 타입: signed int16, big-endian
- 단위:
    Force  = 0.1 N
    Torque = 0.1 Nm
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------
@dataclass
class FTReading:
    """단일 핑거팁 F/T 센서 측정값"""
    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FTFrame:
    """전체 핑거팁 센서 프레임"""
    sensors: Dict[int, FTReading]  # key: sensor_id (1~5)

    def as_dict(self) -> Dict[int, Dict[str, float]]:
        return {sid: reading.as_dict() for sid, reading in self.sensors.items()}


# ---------------------------------------------------------------------
# 예외 클래스
# ---------------------------------------------------------------------
class FTClientError(Exception):
    """F/T 센서 통신 관련 예외"""
    pass


# ---------------------------------------------------------------------
# 클라이언트
# ---------------------------------------------------------------------
class DGFingertipFTClient:
    """
    DG-5F-M 개발자 모드 F/T 센서 전용 TCP 클라이언트

    프로토콜:
      - Get Data: CMD=0x01
      - F/T Sensor code: 0x05
      - 요청 패킷 예: 00 04 01 05
          Length(2) = 4 bytes total
          CMD(1)    = 0x01
          Data(1)   = 0x05  (F/T Sensor)

    응답 패킷:
      Length(2) + CMD(1) + [센서 데이터들]
      센서 데이터는 1개당 12 bytes = 6 * int16
    """

    CMD_GET_DATA = 0x01
    DATA_FT = 0x05
    CMD_SET_FT_OFFSET = 0x0B

    def __init__(
        self,
        host: str,
        port: int = 502,
        timeout: float = 0.5,
        num_sensors: int = 5,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.num_sensors = num_sensors
        self.sock: Optional[socket.socket] = None

    # ---------------------------
    # 연결/해제
    # ---------------------------
    def connect(self) -> bool:
        try:
            self.sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout
            )
            self.sock.settimeout(self.timeout)
            return True
        except OSError as e:
            self.sock = None
            raise FTClientError(f"F/T 센서 클라이언트 연결 실패: {e}") from e

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def ensure_connected(self):
        if self.sock is None:
            raise FTClientError("소켓이 연결되어 있지 않습니다.")

    # ---------------------------
    # 저수준 유틸
    # ---------------------------
    def _recv_exact(self, size: int) -> bytes:
        self.ensure_connected()
        chunks = []
        remaining = size

        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise FTClientError("소켓이 예기치 않게 종료되었습니다.")
            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _send_packet(self, payload: bytes):
        self.ensure_connected()
        try:
            self.sock.sendall(payload)
        except OSError as e:
            raise FTClientError(f"패킷 송신 실패: {e}") from e

    def _recv_packet(self) -> bytes:
        """
        공통 응답:
        [Length(2 bytes)][Rest(length-2 bytes)]
        """
        header = self._recv_exact(2)
        total_len = struct.unpack(">H", header)[0]
        if total_len < 3:
            raise FTClientError(f"비정상 Length 수신: {total_len}")

        body = self._recv_exact(total_len - 2)
        return header + body

    # ---------------------------
    # 프로토콜 명령
    # ---------------------------
    def build_get_ft_packet(self) -> bytes:
        """
        Get Data(0x01) with Data=0x05(F/T)
        전체 길이 = 4 bytes
        """
        total_len = 4
        return struct.pack(">HBB", total_len, self.CMD_GET_DATA, self.DATA_FT)

    def build_set_ft_offset_packet(self) -> bytes:
        """
        Set F/T Sensor Offset(0x0B)
        전체 길이 = 3 bytes
        """
        total_len = 3
        return struct.pack(">HB", total_len, self.CMD_SET_FT_OFFSET)

    # ---------------------------
    # 파싱
    # ---------------------------
    def parse_ft_response(self, packet: bytes) -> FTFrame:
        """
        응답 형식:
          Length(2) + CMD(1) + sensor_data...
        sensor_data는 센서당 12 bytes:
          Fx, Fy, Fz, Tx, Ty, Tz (각각 int16, big-endian)
        단위 환산:
          Force  raw / 10.0  => N
          Torque raw / 10.0  => Nm
        """
        if len(packet) < 3:
            raise FTClientError("응답 길이가 너무 짧습니다.")

        total_len = struct.unpack(">H", packet[:2])[0]
        cmd = packet[2]

        if total_len != len(packet):
            raise FTClientError(
                f"Length 불일치: 헤더={total_len}, 실제={len(packet)}"
            )

        if cmd != self.CMD_GET_DATA:
            raise FTClientError(f"예상하지 못한 CMD 응답: 0x{cmd:02X}")

        payload = packet[3:]
        expected_size = self.num_sensors * 12

        if len(payload) < expected_size:
            raise FTClientError(
                f"F/T payload 크기 부족: 기대={expected_size}, 실제={len(payload)}"
            )

        sensors: Dict[int, FTReading] = {}

        for i in range(self.num_sensors):
            start = i * 12
            chunk = payload[start:start + 12]

            fx_raw, fy_raw, fz_raw, tx_raw, ty_raw, tz_raw = struct.unpack(">hhhhhh", chunk)

            sensors[i + 1] = FTReading(
                fx=fx_raw / 10.0,
                fy=fy_raw / 10.0,
                fz=fz_raw / 10.0,
                tx=tx_raw / 10.0,
                ty=ty_raw / 10.0,
                tz=tz_raw / 10.0,
            )

        return FTFrame(sensors=sensors)

    # ---------------------------
    # 고수준 API
    # ---------------------------
    def read_ft_once(self) -> FTFrame:
        """
        핑거팁 F/T 센서를 1회 읽는다.
        """
        packet = self.build_get_ft_packet()
        self._send_packet(packet)
        resp = self._recv_packet()
        return self.parse_ft_response(resp)

    def set_ft_offset(self):
        """
        현재 상태를 기준으로 F/T 센서 오프셋 설정
        """
        packet = self.build_set_ft_offset_packet()
        self._send_packet(packet)

    def read_ft_dict(self) -> Dict[int, Dict[str, float]]:
        """
        dict 형태로 반환
        {
            1: {"fx": ..., "fy": ..., ...},
            ...
            5: {...}
        }
        """
        return self.read_ft_once().as_dict()
