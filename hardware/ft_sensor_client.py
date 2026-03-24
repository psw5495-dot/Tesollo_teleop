#ft_sensor_client.py
"""
Force/Torque Sensor Client for Tesollo Hand Teleoperation
Real-time 6-axis force/torque data acquisition with safety monitoring
"""

import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict
import logging

from config.constants import (
    FT_SENSOR_IP, FT_SENSOR_PORT, FT_TIMEOUT,
    FT_SAMPLE_RATE, MAX_FORCE_LIMIT, MAX_TORQUE_LIMIT,
    FT_CALIBRATION_MATRIX, FT_FILTER_ALPHA
)


class FTSensorClient:
    """
    Force/Torque 센서와의 실시간 통신을 담당하는 클라이언트
    gripper_client.py와 유사한 구조로 TCP 통신 구현
    """

    def __init__(self, ip: str = FT_SENSOR_IP, port: int = FT_SENSOR_PORT):
        """F/T 센서 클라이언트 초기화"""
        self.ip = ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False

        # 센서 데이터 저장 (gripper_client 패턴 유사)
        self.raw_data = np.zeros(6)  # [Fx, Fy, Fz, Tx, Ty, Tz]
        self.force = np.zeros(3)  # 캘리브레이션 적용된 힘
        self.torque = np.zeros(3)  # 캘리브레이션 적용된 토크

        # 필터링된 데이터 (노이즈 제거용)
        self.filtered_force = np.zeros(3)
        self.filtered_torque = np.zeros(3)

        # 캘리브레이션 및 영점 조정
        self.calibration_matrix = np.array(FT_CALIBRATION_MATRIX)
        self.bias = np.zeros(6)
        self.is_biased = False

        # 스레딩 (실시간 데이터 수집용)
        self.reading_thread: Optional[threading.Thread] = None
        self.stop_reading = threading.Event()
        self.data_lock = threading.Lock()

        # 상태 모니터링
        self.sample_count = 0
        self.error_count = 0
        self.last_update_time = time.time()

        # 로깅 설정
        self.logger = logging.getLogger("FTSensor")

    def connect(self) -> bool:
        """
        F/T 센서에 TCP 연결 (gripper_client.connect() 패턴 유사)

        Returns:
            연결 성공 여부
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(FT_TIMEOUT)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            self.logger.info(f"✓ F/T Sensor connected: {self.ip}:{self.port}")
            return True

        except socket.error as e:
            self.logger.error(f"✗ F/T Sensor connection failed: {e}")
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
            return False

    def disconnect(self):
        """센서 연결 종료 (gripper_client.disconnect() 패턴)"""
        self.stop_reading.set()

        # 읽기 스레드 종료 대기
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=2.0)

        # 소켓 종료
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        self.connected = False
        self.logger.info("✓ F/T Sensor disconnected")

    def start_reading(self) -> bool:
        """
        백그라운드에서 센서 데이터 읽기 시작
        메인 루프와 독립적으로 실시간 데이터 수집
        """
        if not self.connected:
            self.logger.error("Cannot start reading: sensor not connected")
            return False

        self.stop_reading.clear()
        self.reading_thread = threading.Thread(
            target=self._reading_loop,
            daemon=True,
            name="FTSensorReader"
        )
        self.reading_thread.start()
        self.logger.info("✓ F/T Sensor reading started")
        return True

    def _reading_loop(self):
        """
        센서 데이터 연속 읽기 루프 (백그라운드 스레드)
        gripper_client의 명령 전송과 대응되는 데이터 수신 로직
        """
        while not self.stop_reading.is_set() and self.connected:
            try:
                # 센서 데이터 요청 (센서 프로토콜에 따라 수정 필요)
                if hasattr(self, '_send_data_request'):
                    self._send_data_request()

                # 데이터 수신 (일반적으로 6축 * 4바이트 = 24바이트)
                raw_bytes = self._receive_exact_bytes(24)
                if raw_bytes:
                    self._parse_and_update_data(raw_bytes)

                # 샘플링 레이트 제어
                time.sleep(1.0 / FT_SAMPLE_RATE)

            except socket.timeout:
                continue
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"F/T reading error: {e}")

                # 에러가 너무 많으면 중단
                if self.error_count > 20:
                    self.logger.error("Too many F/T sensor errors, stopping")
                    break

    def _receive_exact_bytes(self, size: int) -> Optional[bytes]:
        """
        정확한 바이트 수만큼 수신 (gripper_client 패턴)

        Args:
            size: 수신할 바이트 수

        Returns:
            수신된 데이터 또는 None
        """
        if not self.socket:
            return None

        data = b''
        while len(data) < size:
            try:
                chunk = self.socket.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("F/T sensor connection closed")
                data += chunk
            except socket.timeout:
                return None
            except Exception as e:
                self.logger.error(f"Receive error: {e}")
                return None

        return data

    def _parse_and_update_data(self, data: bytes):
        """
        수신된 바이너리 데이터를 파싱하고 내부 상태 업데이트

        Args:
            data: 센서로부터 받은 원시 데이터 (24바이트)
        """
        try:
            # 6개 float 언패킹 (센서 매뉴얼에 따라 엔디안 수정 필요)
            # '!6f': 빅엔디안, 6개 float
            # '<6f': 리틀엔디안, 6개 float
            raw_values = struct.unpack('!6f', data)

            with self.data_lock:
                self.raw_data = np.array(raw_values)

                # 영점 보정 적용
                if self.is_biased:
                    corrected_data = self.raw_data - self.bias
                else:
                    corrected_data = self.raw_data

                # 캘리브레이션 매트릭스 적용
                calibrated = self.calibration_matrix @ corrected_data

                # 힘과 토크 분리
                self.force = calibrated[:3]
                self.torque = calibrated[3:]

                # 지수 이동 평균 필터 적용
                self.filtered_force = (
                        FT_FILTER_ALPHA * self.force +
                        (1 - FT_FILTER_ALPHA) * self.filtered_force
                )
                self.filtered_torque = (
                        FT_FILTER_ALPHA * self.torque +
                        (1 - FT_FILTER_ALPHA) * self.filtered_torque
                )

                # 상태 업데이트
                self.sample_count += 1
                self.last_update_time = time.time()

        except struct.error as e:
            self.logger.error(f"Data parsing error: {e}")
            self.error_count += 1

    def get_force_torque(self, filtered: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 힘과 토크 값 반환 (gripper_client.get_status() 패턴 유사)

        Args:
            filtered: True면 필터링된 값, False면 원시 값

        Returns:
            (force[3], torque[3]) 튜플
        """
        with self.data_lock:
            if filtered:
                return self.filtered_force.copy(), self.filtered_torque.copy()
            else:
                return self.force.copy(), self.torque.copy()

    def get_magnitude(self) -> Tuple[float, float]:
        """
        힘과 토크의 크기 반환 (안전 체크용)

        Returns:
            (force_magnitude, torque_magnitude)
        """
        with self.data_lock:
            force_mag = np.linalg.norm(self.filtered_force)
            torque_mag = np.linalg.norm(self.filtered_torque)
            return force_mag, torque_mag

    def set_bias(self):
        """
        현재 센서 값을 영점으로 설정 (무부하 상태에서 호출)
        gripper_client의 캘리브레이션 기능과 유사
        """
        with self.data_lock:
            self.bias = self.raw_data.copy()
            self.is_biased = True
            self.logger.info("✓ F/T Sensor bias set (tared)")

    def clear_bias(self):
        """영점 보정 해제"""
        with self.data_lock:
            self.bias = np.zeros(6)
            self.is_biased = False
            self.logger.info("✓ F/T Sensor bias cleared")

    def check_safety_limits(self) -> Tuple[bool, str]:
        """
        안전 제한값 초과 여부 확인

        Returns:
            (초과 여부, 경고 메시지)
        """
        force_mag, torque_mag = self.get_magnitude()

        if force_mag > MAX_FORCE_LIMIT:
            return True, f"Force limit exceeded: {force_mag:.2f}N > {MAX_FORCE_LIMIT}N"

        if torque_mag > MAX_TORQUE_LIMIT:
            return True, f"Torque limit exceeded: {torque_mag:.2f}Nm > {MAX_TORQUE_LIMIT}Nm"

        return False, "Safe"

    def get_connection_status(self) -> Dict:
        """
        연결 상태 및 통계 정보 반환 (gripper_client.get_status() 패턴)

        Returns:
            상태 정보 딕셔너리
        """
        with self.data_lock:
            time_since_update = time.time() - self.last_update_time
            data_rate = self.sample_count / max(time_since_update, 0.001)

            return {
                'connected': self.connected,
                'sample_count': self.sample_count,
                'error_count': self.error_count,
                'is_biased': self.is_biased,
                'data_rate_hz': data_rate,
                'time_since_update': time_since_update
            }


# 센서별 프로토콜 적응을 위한 예시 (실제 센서에 맞게 수정)
class ATIFTSensorClient(FTSensorClient):
    """ATI Force/Torque 센서용 특화 클라이언트"""

    def _send_data_request(self):
        """ATI 센서 데이터 요청 명령"""
        if self.socket:
            # ATI 센서는 보통 연속 스트리밍이므로 별도 요청 불필요
            pass


class RobotiqFTSensorClient(FTSensorClient):
    """Robotiq Force/Torque 센서용 특화 클라이언트"""

    def _send_data_request(self):
        """Robotiq 센서 데이터 요청 명령"""
        if self.socket:
            # Robotiq 특정 프로토콜에 따른 요청 명령
            request_cmd = b'\x01\x04\x00\x00\x00\x06'  # 예시
            self.socket.send(request_cmd)
