#ft_sensor_client.py
"""
Force/Torque Sensor Client with Simulation Mode
Tesollo Hand Teleoperation System
"""

import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict
import logging

# Constants import with fallback
try:
    from config.constants import (
        FT_SENSOR_IP, FT_SENSOR_PORT, FT_TIMEOUT,
        FT_SAMPLE_RATE, MAX_FORCE_LIMIT, MAX_TORQUE_LIMIT,
        FT_CALIBRATION_MATRIX, FT_FILTER_ALPHA, FT_SIMULATION_MODE
    )
except ImportError:
    print("⚠ Warning: Using default F/T sensor constants")
    FT_SENSOR_IP = "192.168.1.100"
    FT_SENSOR_PORT = 49152
    FT_TIMEOUT = 1.0
    FT_SAMPLE_RATE = 100
    MAX_FORCE_LIMIT = 50.0
    MAX_TORQUE_LIMIT = 5.0
    FT_FILTER_ALPHA = 0.2
    FT_SIMULATION_MODE = True
    FT_CALIBRATION_MATRIX = np.eye(6).tolist()


class FTSensorClient:
    """Force/Torque 센서 통신 클라이언트 (시뮬레이션 모드 포함)"""
    
    def __init__(self, ip: str = FT_SENSOR_IP, port: int = FT_SENSOR_PORT):
        self.ip = ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.simulation_mode = FT_SIMULATION_MODE
        
        # 센서 데이터 저장소
        self.raw_data = np.zeros(6)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.filtered_force = np.zeros(3)
        self.filtered_torque = np.zeros(3)
        
        # 캘리브레이션
        self.calibration_matrix = np.array(FT_CALIBRATION_MATRIX)
        self.bias = np.zeros(6)
        self.is_biased = False
        
        # 스레딩 제어
        self.reading_thread: Optional[threading.Thread] = None
        self.stop_reading = threading.Event()
        self.data_lock = threading.Lock()
        
        # 통계
        self.sample_count = 0
        self.error_count = 0
        self.last_update_time = time.time()
        self.start_time = time.time()
        
        # 시뮬레이션용
        self._sim_time = 0.0
        
        self.logger = logging.getLogger("FTSensor")
    
    def connect(self) -> bool:
        """F/T 센서 연결 (시뮬레이션 모드 자동 전환)"""
        if self.simulation_mode:
            self.connected = True
            self.logger.info("✓ F/T Sensor: Simulation mode enabled")
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(FT_TIMEOUT)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            self.logger.info(f"✓ F/T Sensor connected: {self.ip}:{self.port}")
            return True
        except socket.error as e:
            self.logger.warning(f"⚠ Physical sensor connection failed: {e}")
            self.logger.info("→ Switching to simulation mode")
            self.simulation_mode = True
            self.connected = True
            return True
    
    def disconnect(self):
        """센서 연결 종료"""
        self.stop_reading.set()
        
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=2.0)
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.connected = False
        self.logger.info("✓ F/T Sensor disconnected")
    
    def start_reading(self) -> bool:
        """백그라운드 데이터 읽기 시작"""
        if not self.connected:
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
        """센서 데이터 연속 읽기 루프"""
        while not self.stop_reading.is_set() and self.connected:
            try:
                if self.simulation_mode:
                    self._generate_simulation_data()
                else:
                    raw_bytes = self._receive_exact_bytes(24)
                    if raw_bytes:
                        self._parse_and_update_data(raw_bytes)
                
                time.sleep(1.0 / FT_SAMPLE_RATE)
                
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Reading error: {e}")
                if self.error_count > 20:
                    break
    
    def _generate_simulation_data(self):
        """시뮬레이션 데이터 생성 (현실적인 F/T 패턴)"""
        self._sim_time += 1.0 / FT_SAMPLE_RATE
        
        # 사인파 + 노이즈로 현실적인 F/T 데이터 생성
        force_sim = np.array([
            15 * np.sin(self._sim_time * 0.5) + np.random.normal(0, 1),
            10 * np.cos(self._sim_time * 0.3) + np.random.normal(0, 0.8),
            20 + 8 * np.sin(self._sim_time * 0.2) + np.random.normal(0, 0.5)
        ])
        
        torque_sim = np.array([
            2 * np.sin(self._sim_time * 0.7) + np.random.normal(0, 0.2),
            1.5 * np.cos(self._sim_time * 0.4) + np.random.normal(0, 0.15),
            1 + 0.8 * np.sin(self._sim_time * 0.6) + np.random.normal(0, 0.1)
        ])
        
        # 데이터 업데이트
        with self.data_lock:
            self.raw_data = np.concatenate([force_sim, torque_sim])
            
            if self.is_biased:
                corrected = self.raw_data - self.bias
            else:
                corrected = self.raw_data
            
            calibrated = self.calibration_matrix @ corrected
            self.force = calibrated[:3]
            self.torque = calibrated[3:]
            
            # 지수 이동 평균 필터 적용
            self.filtered_force = (FT_FILTER_ALPHA * self.force + 
                                  (1 - FT_FILTER_ALPHA) * self.filtered_force)
            self.filtered_torque = (FT_FILTER_ALPHA * self.torque + 
                                   (1 - FT_FILTER_ALPHA) * self.filtered_torque)
            
            self.sample_count += 1
            self.last_update_time = time.time()
    
    def _receive_exact_bytes(self, size: int) -> Optional[bytes]:
        """정확한 바이트 수 수신"""
        if not self.socket:
            return None
        
        data = b''
        while len(data) < size:
            try:
                chunk = self.socket.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            except socket.timeout:
                return None
        return data
    
    def _parse_and_update_data(self, data: bytes):
        """실제 센서 데이터 파싱"""
        try:
            raw_values = struct.unpack('!6f', data)  # 빅엔디안 6개 float
            
            with self.data_lock:
                self.raw_data = np.array(raw_values)
                
                if self.is_biased:
                    corrected = self.raw_data - self.bias
                else:
                    corrected = self.raw_data
                
                calibrated = self.calibration_matrix @ corrected
                self.force = calibrated[:3]
                self.torque = calibrated[3:]
                
                self.filtered_force = (FT_FILTER_ALPHA * self.force + 
                                      (1 - FT_FILTER_ALPHA) * self.filtered_force)
                self.filtered_torque = (FT_FILTER_ALPHA * self.torque + 
                                       (1 - FT_FILTER_ALPHA) * self.filtered_torque)
                
                self.sample_count += 1
                self.last_update_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            self.error_count += 1
    
    def get_force_torque(self, filtered: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """현재 힘/토크 값 반환"""
        with self.data_lock:
            if filtered:
                return self.filtered_force.copy(), self.filtered_torque.copy()
            else:
                return self.force.copy(), self.torque.copy()
    
    def get_magnitude(self) -> Tuple[float, float]:
        """힘/토크 크기 반환"""
        with self.data_lock:
            force_mag = np.linalg.norm(self.filtered_force)
            torque_mag = np.linalg.norm(self.filtered_torque)
            return force_mag, torque_mag
    
    def set_bias(self):
        """영점 설정 (현재 값을 0점으로)"""
        with self.data_lock:
            self.bias = self.raw_data.copy()
            self.is_biased = True
            self.logger.info("✓ F/T Sensor bias set (tared)")
    
    def clear_bias(self):
        """영점 해제"""
        with self.data_lock:
            self.bias = np.zeros(6)
            self.is_biased = False
    
    def check_safety_limits(self) -> Tuple[bool, str]:
        """안전 제한 확인"""
        force_mag, torque_mag = self.get_magnitude()
        
        if force_mag > MAX_FORCE_LIMIT:
            return True, f"Force limit exceeded: {force_mag:.2f}N > {MAX_FORCE_LIMIT}N"
        
        if torque_mag > MAX_TORQUE_LIMIT:
            return True, f"Torque limit exceeded: {torque_mag:.2f}Nm > {MAX_TORQUE_LIMIT}Nm"
        
        return False, "Safe"
    
    def get_connection_status(self) -> Dict:
        """연결 상태 및 통계 반환"""
        with self.data_lock:
            runtime = time.time() - self.start_time
            data_rate = self.sample_count / max(runtime, 1)
            
            return {
                'connected': self.connected,
                'simulation_mode': self.simulation_mode,
                'sample_count': self.sample_count,
                'error_count': self.error_count,
                'is_biased': self.is_biased,
                'data_rate_hz': data_rate,
                'runtime': runtime
            }
