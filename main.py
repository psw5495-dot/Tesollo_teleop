"""
Tesollo Hand Teleoperation System with F/T Sensor Integration
"""

import cv2
import time
import numpy as np
import sys
import os

# 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Hardware imports
try:
    from hardware.gripper_client import GripperClient
    from hardware.ft_sensor_client import FTSensorClient  # 수정된 경로
except ImportError as e:
    print(f"❌ Hardware import error: {e}")
    sys.exit(1)

# Vision imports
try:
    from vision.hand_tracker import HandTracker
except ImportError as e:
    print(f"❌ Vision import error: {e}")
    sys.exit(1)

# Control imports
try:
    from control.motor_controller import MotorController
    from control.safety import SafetyController
except ImportError as e:
    print(f"❌ Control import error: {e}")
    sys.exit(1)

# UI imports
try:
    from ui.visualization import Visualizer
    from ui.ft_dashboard import FTDashboard
except ImportError as e:
    print(f"❌ UI import error: {e}")
    sys.exit(1)

# Constants
try:
    from config.constants import (
        CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
        MAX_FORCE_LIMIT, MAX_TORQUE_LIMIT, LOOP_RATE
    )
except ImportError as e:
    print(f"❌ Constants import error: {e}")
    # 기본값 사용
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    LOOP_RATE = 1/30
    MAX_FORCE_LIMIT = 50.0
    MAX_TORQUE_LIMIT = 5.0


class TeleopSystem:
    """Tesollo Hand Teleoperation System with F/T Sensor"""
    
    def __init__(self):
        print("="*60)
        print("TESOLLO HAND TELEOPERATION SYSTEM")
        print("with Force/Torque Sensor Integration")
        print("="*60)
        
        self.running = True
        self.emergency_stop = False
        self.ft_sensor_enabled = False
        
        # 컴포넌트 초기화
        print("\n[1/7] Initializing camera...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print("❌ Failed to open camera")
            sys.exit(1)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        print("✓ Camera initialized")
        
        print("\n[2/7] Initializing hand tracker...")
        self.hand_tracker = HandTracker()
        print("✓ Hand tracker initialized")
        
        print("\n[3/7] Initializing motor controller...")
        self.motor_controller = MotorController()
        print("✓ Motor controller initialized")
        
        print("\n[4/7] Initializing safety controller...")
        self.safety = SafetyController()
        print("✓ Safety controller initialized")
        
        print("\n[5/7] Initializing visualizer...")
        self.visualizer = Visualizer()
        print("✓ Visualizer initialized")
        
        print("\n[6/7] Initializing gripper client...")
        self.gripper = GripperClient()
        print("✓ Gripper client initialized")
        
        print("\n[7/7] Initializing F/T sensor system...")
        try:
            self.ft_sensor = FTSensorClient()
            self.ft_dashboard = FTDashboard(width=1000, height=700)
            print("✓ F/T sensor system initialized")
        except Exception as e:
            print(f"⚠ F/T sensor initialization warning: {e}")
            self.ft_sensor = None
            self.ft_dashboard = None
        
        print("\n✓ All systems initialized\n")
    
    def initialize_hardware(self) -> bool:
        """하드웨어 연결 및 초기화"""
        print("="*60)
        print("HARDWARE CONNECTION")
        print("="*60)
        
        # Gripper 연결
        print("\n[1/2] Connecting to gripper...")
        if not self.gripper.connect():
            print("⚠ Gripper connection failed (continuing anyway)")
        else:
            print("✓ Gripper connected")
        
        # F/T 센서 연결
        print("\n[2/2] Connecting to F/T sensor...")
        if self.ft_sensor:
            try:
                if self.ft_sensor.connect():
                    if self.ft_sensor.start_reading():
                        print("✓ F/T sensor connected and reading started")
                        
                        # 안정화 대기
                        print("  Waiting for sensor stabilization...")
                        time.sleep(2.0)
                        
                        # 영점 설정
                        self.ft_sensor.set_bias()
                        print("  ✓ Sensor bias set (tared)")
                        
                        self.ft_sensor_enabled = True
                        
                        # 센서 상태 확인
                        status = self.ft_sensor.get_connection_status()
                        if status['simulation_mode']:
                            print("  ℹ Running in SIMULATION mode")
                        else:
                            print(f"  ℹ Connected to physical sensor at {self.ft_sensor.ip}")
                    else:
                        print("⚠ F/T sensor connected but reading failed")
                else:
                    print("⚠ F/T sensor connection failed")
            except Exception as e:
                print(f"⚠ F/T sensor error: {e}")
        
        print("\n✓ Hardware initialization complete\n")
        return True
    
    def main_control_loop(self):
        """메인 제어 루프"""
        print("="*60)
        print("STARTING TELEOPERATION")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit system")
        print("  Z - Re-bias F/T sensor")
        print("  R - Reset emergency stop")
        print("  S - Print system status")
        print("="*60 + "\n")
        
        while self.running:
            loop_start = time.time()
            
            # 1. 카메라 프레임 읽기
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # 2. 손 추적
            hand_data = self.hand_tracker.process(frame)
            
            # 3. F/T 센서 데이터 처리
            ft_safe = True
            if self.ft_sensor_enabled and self.ft_sensor:
                try:
                    # 센서 데이터 가져오기
                    force, torque = self.ft_sensor.get_force_torque(filtered=True)
                    
                    # 대시보드 업데이트
                    if self.ft_dashboard:
                        self.ft_dashboard.update_data(force, torque)
                    
                    # 안전 체크
                    exceeded, safety_msg = self.ft_sensor.check_safety_limits()
                    if exceeded and not self.emergency_stop:
                        print(f"🚨 SAFETY ALERT: {safety_msg}")
                        self.emergency_stop = True
                        ft_safe = False
                        
                        # 그리퍼 정지
                        if self.gripper.connected:
                            self.gripper.stop_all_motors()
                
                except Exception as e:
                    print(f"⚠ F/T processing error: {e}")
            
            # 4. 모터 제어 (안전할 때만)
            if hand_data and ft_safe and not self.emergency_stop:
                try:
                    motor_commands = self.motor_controller.compute_commands(hand_data)
                    safe_commands = self.safety.validate_commands(motor_commands)
                    
                    if self.gripper.connected:
                        self.gripper.send_commands(safe_commands)
                except Exception as e:
                    print(f"⚠ Motor control error: {e}")
            
            # 5. 화면 표시 (핵심: 모든 imshow를 메인 스레드에서 처리)
            try:
                # 메인 카메라 화면
                self._draw_main_overlay(frame, hand_data)
                cv2.imshow("Tesollo Hand Teleoperation", frame)
                
                # F/T 대시보드 (별도 창)
                if self.ft_dashboard:
                    dashboard_frame = self.ft_dashboard.get_dashboard_frame()
                    cv2.imshow("F/T Sensor Dashboard", dashboard_frame)
                    
            except Exception as e:
                print(f"⚠ Display error: {e}")
            
            # 6. 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⚠ Quit requested by user")
                break
            elif key == ord('z') and self.ft_sensor_enabled:
                self.ft_sensor.set_bias()
                print("✓ F/T sensor re-biased")
            elif key == ord('r'):
                if self.emergency_stop:
                    self.emergency_stop = False
                    print("✓ Emergency stop reset")
            elif key == ord('s'):
                self._print_system_status()
            
            # 7. 루프 타이밍 (30fps 유지)
            loop_time = time.time() - loop_start
            if loop_time < LOOP_RATE:
                time.sleep(LOOP_RATE - loop_time)
    
    def _draw_main_overlay(self, frame, hand_data):
        """메인 화면 오버레이"""
        h, w = frame.shape[:2]
        
        # 반투명 상태 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # F/T 센서 상태 표시
        if self.ft_sensor_enabled:
            force_mag, torque_mag = self.ft_sensor.get_magnitude()
            
            if self.emergency_stop:
                status_color = (0, 0, 255)  # Red
                status_text = "E-STOP"
            elif force_mag > MAX_FORCE_LIMIT * 0.8:
                status_color = (0, 255, 255)  # Yellow
                status_text = "WARNING"
            else:
                status_color = (0, 255, 0)  # Green
                status_text = "ACTIVE"
            
            cv2.putText(frame, f"F/T: {status_text}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 현재 값 표시
            cv2.putText(frame, f"F:{force_mag:.1f}N T:{torque_mag:.2f}Nm", 
                       (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "F/T: OFFLINE", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    def _print_system_status(self):
        """시스템 상태 출력"""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        
        if self.ft_sensor_enabled:
            status = self.ft_sensor.get_connection_status()
            force_mag, torque_mag = self.ft_sensor.get_magnitude()
            
            print(f"\nF/T Sensor: ✓ Active")
            print(f"  Mode: {'Simulation' if status['simulation_mode'] else 'Physical'}")
            print(f"  Force: {force_mag:.2f}N (Max: {MAX_FORCE_LIMIT}N)")
            print(f"  Torque: {torque_mag:.3f}Nm (Max: {MAX_TORQUE_LIMIT}Nm)")
            print(f"  Sample Rate: {status['data_rate_hz']:.1f} Hz")
        else:
            print(f"\nF/T Sensor: ✗ Not Available")
        
        print(f"\nGripper: {'✓ Connected' if self.gripper.connected else '✗ Disconnected'}")
        print(f"Emergency Stop: {'🚨 ACTIVE' if self.emergency_stop else '✓ Inactive'}")
        print("="*60 + "\n")
    
    def shutdown(self):
        """시스템 종료"""
        print("\n" + "="*60)
        print("SHUTTING DOWN SYSTEM")
        print("="*60)
        
        if self.ft_sensor_enabled and self.ft_sensor:
            print("\nDisconnecting F/T sensor...")
            self.ft_sensor.disconnect()
        
        if self.gripper and self.gripper.connected:
            print("Disconnecting gripper...")
            self.gripper.disconnect()
        
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        print("\n✓ System shutdown complete")


def main():
    """메인 실행 함수"""
    system = None
    
    try:
        system = TeleopSystem()
        
        if system.initialize_hardware():
            system.main_control_loop()
        else:
            print("❌ Hardware initialization failed")
            
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.shutdown()


if __name__ == "__main__":
    main()
