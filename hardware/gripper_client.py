#gripper_client.py
"""
DG-5F-M 개발자 모드 TCP 클라이언트
"""
import socket
import struct
import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DG5FDevClient:
    """DG-5F-M 그리퍼 개발자 모드 TCP 클라이언트"""

    def __init__(self, ip: str, port: int, timeout: float = 0.5):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.connected = False

    def connect(self) -> bool:
        """TCP 연결 설정"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            self.connected = True
            logger.info(f"그리퍼 연결 성공: {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"그리퍼 연결 실패: {e}")
            self.connected = False
            return False

    def close(self):
        """TCP 연결 종료"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.warning(f"소켓 종료 중 오류: {e}")
            finally:
                self.sock = None
                self.connected = False

    def _recv_exact(self, n: int) -> bytes:
        """정확히 n바이트 수신"""
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("소켓이 닫혔습니다")
                buf += chunk
            except socket.timeout:
                raise TimeoutError("데이터 수신 타임아웃")
        return buf

    def send_only(self, cmd: int, data: bytes = b""):
        """응답 없이 명령 전송"""
        if not self.connected or not self.sock:
            raise ConnectionError("그리퍼가 연결되지 않았습니다")

        length = 2 + 1 + len(data)
        packet = struct.pack(">H", length) + struct.pack("B", cmd) + data
        self.sock.sendall(packet)

    def transact(self, cmd: int, data: bytes = b"") -> bytes:
        """명령 전송 및 응답 수신"""
        if not self.connected or not self.sock:
            raise ConnectionError("그리퍼가 연결되지 않았습니다")

        length = 2 + 1 + len(data)
        packet = struct.pack(">H", length) + struct.pack("B", cmd) + data
        self.sock.sendall(packet)

        resp_len = struct.unpack(">H", self._recv_exact(2))[0]
        resp_rest = self._recv_exact(resp_len - 2)
        return resp_rest

    def get_positions(self) -> Dict[int, int]:
        """현재 모터 위치 획득 (0.1도 단위)"""
        try:
            resp = self.transact(0x01, data=bytes([0x01]))
            if not resp or resp[0] != 0x01:
                raise RuntimeError(f"예상치 못한 응답 CMD: {resp[0] if resp else None}")

            payload = resp[1:]
            positions = {}
            i = 0
            while i + 3 <= len(payload):
                joint_id = payload[i]
                value = struct.unpack(">h", payload[i + 1:i + 3])[0]
                positions[joint_id] = value
                i += 3

            return positions
        except Exception as e:
            logger.error(f"위치 읽기 실패: {e}")
            raise

    def set_duty(self, duty_by_id: Dict[int, int]):
        """모든 모터의 듀티 사이클 설정"""
        try:
            data = b""
            for joint_id in range(1, 21):
                duty = int(np.clip(int(duty_by_id.get(joint_id, 0)), -1000, 1000))
                data += struct.pack("B", joint_id) + struct.pack(">h", duty)
            self.send_only(0x05, data=data)
        except Exception as e:
            logger.error(f"듀티 설정 실패: {e}")
            raise

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_ft_data(self) -> Dict[int, Dict[str, float]]:
        """F/T 센서 데이터 읽기 (통합 버전)"""
        try:
            # F/T 센서 요청 패킷 (CMD=0x01, Data=0x05)
            resp = self.transact(0x01, data=bytes([0x05]))

            if not resp or resp[0] != 0x01:
                raise RuntimeError(f"Unexpected F/T response CMD: {resp[0] if resp else None}")

            payload = resp[1:]
            sensors = {}

            # 5개 센서, 각각 12바이트 (6개 int16)
            for i in range(5):
                start = i * 12
                if start + 12 <= len(payload):
                    chunk = payload[start:start + 12]
                    fx_raw, fy_raw, fz_raw, tx_raw, ty_raw, tz_raw = struct.unpack(">hhhhhh", chunk)

                    sensors[i + 1] = {
                        'fx': fx_raw / 10.0,  # 0.1N → N
                        'fy': fy_raw / 10.0,
                        'fz': fz_raw / 10.0,
                        'tx': tx_raw / 10.0,  # 0.1Nm → Nm
                        'ty': ty_raw / 10.0,
                        'tz': tz_raw / 10.0,
                    }

            return sensors

        except Exception as e:
            logger.error(f"F/T data read failed: {e}")
            return {}

    def set_ft_offset(self):
        """F/T 센서 오프셋 설정"""
        try:
            self.send_only(0x0B)  # Set F/T Offset 명령
            logger.info("F/T sensor offset set")
        except Exception as e:
            logger.error(f"F/T offset set failed: {e}")
            raise
