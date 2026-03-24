"""
Tesollo Hand Teleoperation System with F/T Sensor Integration
Error-Safe Version with Robust Initialization
"""

import cv2
import time
import numpy as np
import sys
import os

# 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TeleopSystem:
    """Tesollo Hand Teleoperation System with F/T Sensor"""

    def __init__(self):
        """안전한 초기화 (에러 방지)"""
        print("="*60)
        print("TESOLLO HAND TELEOPERATION SYSTEM")
        print("with Force/Torque Sensor Integration")
        print("="*60)

        # 초기화 상태 관리
        self.initialized = False
        self.running = True
        self.emergency_stop = False
        self.ft_sensor_enabled = False

        # [핵심 안전장치] 모든 속성을 먼저 None으로 초기화
        self.camera = None
        self.hand_tracker = None
        self.motor_controller = None
        self.safety = None
        self.visualizer = None
        self.gripper = None
        self.ft_sensor = None
        self.ft_dashboard = None

        # 기본 설정값 (constants.py 없어도 작동)
        self.camera_index = 0
        self.camera_width = 640
        self.camera_height = 480
        self.loop_rate = 1/30

        try:
            from config.constants import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, LOOP_RATE
            self.camera_index = CAMERA_INDEX
            self.camera_width = CAMERA_WIDTH
            self.camera_height = CAMERA_HEIGHT
            self.loop_rate = LOOP_RATE
        except ImportError:
            print("⚠ Using default camera settings")

        # 각 컴포넌트를 개별적으로 안전하게 초기화
        self._initialize_components()

    def _initialize_components(self):
        """컴포넌트별 안전한 초기화"""

        # 1. 카메라 초기화 (필수 컴포넌트)
        print("\n[1/7] Initializing camera...")
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                # 인덱스 0 실패시 1번 시도
                print("  Trying camera index 1...")
                self.camera = cv2.VideoCapture(1)
                if not self.camera.isOpened():
                    raise RuntimeError("No camera available")

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            print("✓ Camera initialized")

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("  System cannot continue without camera")
            return

        # 2. Hand Tracker 초기화 (선택적)
        print("\n[2/7] Initializing hand tracker...")
        try:
            from vision.hand_tracker import HandTracker
            self.hand_tracker = HandTracker()
            print("✓ Hand tracker initialized")
        except Exception as e:
            print(f"⚠ Hand tracker initialization failed: {e}")
            print("  → Continuing without hand tracking")

        # 3. Motor Controller 초기화 (선택적)
        print("\n[3/7] Initializing motor controller...")
        try:
            from control.motor_controller import MotorController
            self.motor_controller = MotorController()
            print("✓ Motor controller initialized")
        except Exception as e:
            print(f"⚠ Motor controller initialization failed: {e}")
            print("  → Continuing without motor control")

        # 4. Safety Controller 초기화 (선택적)
        print("\n[4/7] Initializing safety controller...")
        try:
            from control.safety import SafetyController
            self.safety = SafetyController()
            print("✓ Safety controller initialized")
        except Exception as e:
            print(f"⚠ Safety controller initialization failed: {e}")
            print("  → Continuing without safety checks")

        # 5. Visualizer 초기화 (선택적)
        print("\n[5/7] Initializing visualizer...")
        try:
            from ui.visualization import Visualizer
            self.visualizer = Visualizer()
            print("✓ Visualizer initialized")
        except Exception as e:
            print(f"⚠ Visualizer initialization failed: {e}")
            print("  → Continuing with basic visualization")

        # 6. Gripper Client 초기화 (선택적)
        print("\n[6/7] Initializing gripper client...")
        try:
            # DG5FDevClient import 처리
            try:
                from hardware.gripper_client import DG5FDevClient as GripperClient
            except ImportError:
                from hardware.gripper_client import GripperClient

            self.gripper = GripperClient()
            print("✓ Gripper client initialized")
        except Exception as e:
            print(f"⚠ Gripper client initialization failed: {e}")
            print("  → Continuing without gripper control")

        # 7. F/T Sensor 시스템 초기화 (선택적)
        print("\n[7/7] Initializing F/T sensor system...")
        try:
            from hardware.ft_sensor_client import FTSensorClient
            from ui.ft_dashboard import FTDashboard

            self.ft_sensor = FTSensorClient()
            self.ft_dashboard = FTDashboard(width=1000, height=700)
            print("✓ F/T sensor system initialized")
        except Exception as e:
            print(f"⚠ F/T sensor system initialization failed: {e}")
            print("  → Continuing without F/T sensor")

        # 초기화 완료 표시
        self.initialized = True
        print("\n✓ System initialization complete")

        # 사용 가능한 기능 요약
        self._print_available_features()

    def _print_available_features(self):
        """초기화된 기능 요약 출력"""
        print("\n" + "-"*40)
        print("AVAILABLE FEATURES:")
        print("-"*40)
        print(f"Camera: {'✓' if self.camera and self.camera.isOpened() else '✗'}")
        print(f"Hand Tracking: {'✓' if self.hand_tracker else '✗'}")
        print(f"Motor Control: {'✓' if self.motor_controller else '✗'}")
        print(f"Safety System: {'✓' if self.safety else '✗'}")
        print(f"Gripper: {'✓' if self.gripper else '✗'}")
        print(f"F/T Sensor: {'✓' if self.ft_sensor else '✗'}")
        print(f"F/T Dashboard: {'✓' if self.ft_dashboard else '✗'}")
        print("-"*40)

    def initialize_hardware(self) -> bool:
        """하드웨어 연결"""
        if not self.initialized:
            print("❌ System not properly initialized")
            return False

        print("\n" + "="*60)
        print("HARDWARE CONNECTION")
        print("="*60)

        # Gripper 연결
        if self.gripper:
            print("\nConnecting to gripper...")
            try:
                if hasattr(self.gripper, 'connect') and self.gripper.connect():
                    print("✓ Gripper connected")
                else:
                    print("⚠ Gripper connection failed (continuing in simulation)")
            except Exception as e:
                print(f"⚠ Gripper connection error: {e}")

        # F/T Sensor 연결
        if self.ft_sensor:
            print("\nConnecting to F/T sensor...")
            try:
                if self.ft_sensor.connect():
                    if hasattr(self.ft_sensor, 'start_reading') and self.ft_sensor.start_reading():
                        print("✓ F/T sensor connected and reading started")

                        print("  Waiting for sensor stabilization...")
                        time.sleep(2.0)

                        if hasattr(self.ft_sensor, 'set_bias'):
                            self.ft_sensor.set_bias()
                            print("  ✓ Sensor bias set (tared)")

                        self.ft_sensor_enabled = True

                        # 연결 상태 확인
                        if hasattr(self.ft_sensor, 'get_connection_status'):
                            status = self.ft_sensor.get_connection_status()
                            mode = "Simulation" if status.get('simulation_mode', False) else "Physical"
                            print(f"  ℹ Running in {mode} mode")
                    else:
                        print("⚠ F/T sensor reading failed")
                else:
                    print("⚠ F/T sensor connection failed")
            except Exception as e:
                print(f"⚠ F/T sensor error: {e}")

        print("\n✓ Hardware initialization complete\n")
        return True

    def main_control_loop(self):
        """메인 제어 루프 (완전 안전 처리)"""
        if not self.initialized:
            print("❌ Cannot start control loop: system not initialized")
            return

        print("="*60)
        print("STARTING TELEOPERATION")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit system")
        print("  Z - Re-bias F/T sensor (if available)")
        print("  R - Reset emergency stop")
        print("  S - Print system status")
        print("="*60 + "\n")

        while self.running:
            try:
                loop_start = time.time()

                # 1. 카메라 프레임 읽기
                if not self.camera or not self.camera.isOpened():
                    print("❌ Camera not available")
                    break

                ret, frame = self.camera.read()
                if not ret:
                    continue

                # 2. 손 추적 (안전 처리)
                hand_data = None
                if self.hand_tracker:
                    try:
                        hand_data = self.hand_tracker.process(frame)
                    except Exception as e:
                        print(f"⚠ Hand tracking error: {e}")

                # 3. F/T 센서 데이터 처리
                ft_safe = True
                if self.ft_sensor_enabled and self.ft_sensor:
                    try:
                        if hasattr(self.ft_sensor, 'get_force_torque'):
                            force, torque = self.ft_sensor.get_force_torque(filtered=True)

                            # 대시보드 업데이트
                            if self.ft_dashboard and hasattr(self.ft_dashboard, 'update_data'):
                                self.ft_dashboard.update_data(force, torque)

                            # 안전 체크
                            if hasattr(self.ft_sensor, 'check_safety_limits'):
                                exceeded, safety_msg = self.ft_sensor.check_safety_limits()
                                if exceeded and not self.emergency_stop:
                                    print(f"🚨 SAFETY ALERT: {safety_msg}")
                                    self.emergency_stop = True
                                    ft_safe = False

                                    # 그리퍼 비상 정지
                                    if self.gripper and hasattr(self.gripper, 'stop_all_motors'):
                                        try:
                                            self.gripper.stop_all_motors()
                                        except Exception:
                                            pass

                    except Exception as e:
                        print(f"⚠ F/T processing error: {e}")

                # 4. 모터 제어 (안전할 때만)
                if hand_data and ft_safe and not self.emergency_stop:
                    try:
                        if self.motor_controller and hasattr(self.motor_controller, 'compute_commands'):
                            motor_commands = self.motor_controller.compute_commands(hand_data)

                            # 안전 검증
                            if self.safety and hasattr(self.safety, 'validate_commands'):
                                safe_commands = self.safety.validate_commands(motor_commands)
                            else:
                                safe_commands = motor_commands

                            # 그리퍼 명령 전송
                            if self.gripper and hasattr(self.gripper, 'send_commands'):
                                self.gripper.send_commands(safe_commands)
                    except Exception as e:
                        print(f"⚠ Motor control error: {e}")

                # 5. 화면 표시
                try:
                    # 메인 화면 오버레이
                    self._draw_main_overlay(frame, hand_data)
                    cv2.imshow("Tesollo Hand Teleoperation", frame)

                    # F/T 대시보드 (별도 창)
                    if self.ft_dashboard and hasattr(self.ft_dashboard, 'get_dashboard_frame'):
                        dashboard_frame = self.ft_dashboard.get_dashboard_frame()
                        cv2.imshow("F/T Sensor Dashboard", dashboard_frame)

                except Exception as e:
                    print(f"⚠ Display error: {e}")

                # 6. 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ Quit requested by user")
                    break
                elif key == ord('z') and self.ft_sensor_enabled and self.ft_sensor:
                    if hasattr(self.ft_sensor, 'set_bias'):
                        self.ft_sensor.set_bias()
                        print("✓ F/T sensor re-biased")
                elif key == ord('r'):
                    if self.emergency_stop:
                        self.emergency_stop = False
                        print("✓ Emergency stop reset")
                elif key == ord('s'):
                    self._print_system_status()

                # 7. 루프 타이밍 유지
                loop_time = time.time() - loop_start
                if loop_time < self.loop_rate:
                    time.sleep(self.loop_rate - loop_time)

            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
            except Exception as e:
                print(f"⚠ Loop error: {e}")

    def _draw_main_overlay(self, frame, hand_data):
        """메인 화면 상태 오버레이"""
        try:
            h, w = frame.shape[:2]

            # 반투명 상태 바
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # F/T 센서 상태
            if self.ft_sensor_enabled and self.ft_sensor:
                try:
                    if hasattr(self.ft_sensor, 'get_magnitude'):
                        force_mag, torque_mag = self.ft_sensor.get_magnitude()

                        if self.emergency_stop:
                            status_color = (0, 0, 255)  # Red
                            status_text = "E-STOP"
                        elif force_mag > 40.0:  # Warning threshold
                            status_color = (0, 255, 255)  # Yellow
                            status_text = "WARNING"
                        else:
                            status_color = (0, 255, 0)  # Green
                            status_text = "ACTIVE"

                        cv2.putText(frame, f"F/T: {status_text}", (10, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                        cv2.putText(frame, f"F:{force_mag:.1f}N T:{torque_mag:.2f}Nm",
                                   (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    else:
                        cv2.putText(frame, "F/T: CONNECTED", (10, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception:
                    cv2.putText(frame, "F/T: ERROR", (10, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "F/T: OFFLINE", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            # 손 추적 상태
            if hand_data:
                cv2.putText(frame, "Hand: DETECTED", (w - 200, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Hand: NO DETECT", (w - 200, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        except Exception:
            pass  # 오버레이 실패해도 메인 기능에 영향 없음

    def _print_system_status(self):
        """상세 시스템 상태 출력"""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)

        print(f"\nInitialization: {'✓ Complete' if self.initialized else '✗ Failed'}")
        print(f"Camera: {'✓ Active' if self.camera and self.camera.isOpened() else '✗ Inactive'}")
        print(f"Hand Tracker: {'✓ Available' if self.hand_tracker else '✗ Not Available'}")
        print(f"Motor Controller: {'✓ Available' if self.motor_controller else '✗ Not Available'}")
        print(f"Safety Controller: {'✓ Available' if self.safety else '✗ Not Available'}")
        print(f"Gripper: {'✓ Available' if self.gripper else '✗ Not Available'}")

        if self.ft_sensor_enabled and self.ft_sensor:
            try:
                if hasattr(self.ft_sensor, 'get_connection_status'):
                    status = self.ft_sensor.get_connection_status()
                    print(f"\nF/T Sensor: ✓ Active")
                    print(f"  Mode: {'Simulation' if status.get('simulation_mode', False) else 'Physical'}")

                    if hasattr(self.ft_sensor, 'get_magnitude'):
                        force_mag, torque_mag = self.ft_sensor.get_magnitude()
                        print(f"  Force: {force_mag:.2f}N")
                        print(f"  Torque: {torque_mag:.3f}Nm")

                    print(f"  Sample Rate: {status.get('data_rate_hz', 0):.1f} Hz")
                else:
                    print(f"\nF/T Sensor: ✓ Connected")
            except Exception:
                print(f"\nF/T Sensor: ⚠ Connection Error")
        else:
            print(f"\nF/T Sensor: ✗ Not Available")

        print(f"\nEmergency Stop: {'🚨 ACTIVE' if self.emergency_stop else '✓ Inactive'}")
        print("="*60 + "\n")

    def shutdown(self):
        """안전한 시스템 종료"""
        print("\n" + "="*60)
        print("SHUTTING DOWN SYSTEM")
        print("="*60)

        # F/T 센서 종료
        if hasattr(self, 'ft_sensor') and self.ft_sensor:
            try:
                print("Disconnecting F/T sensor...")
                if hasattr(self.ft_sensor, 'disconnect'):
                    self.ft_sensor.disconnect()
            except Exception as e:
                print(f"⚠ F/T sensor disconnect error: {e}")

        # 그리퍼 종료
        if hasattr(self, 'gripper') and self.gripper:
            try:
                print("Disconnecting gripper...")
                if hasattr(self.gripper, 'disconnect'):
                    self.gripper.disconnect()
            except Exception as e:
                print(f"⚠ Gripper disconnect error: {e}")

        # 카메라 종료
        if hasattr(self, 'camera') and self.camera:
            try:
                self.camera.release()
            except Exception as e:
                print(f"⚠ Camera release error: {e}")

        # OpenCV 창 닫기
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"⚠ Window close error: {e}")

        print("\n✓ System shutdown complete")


def main():
    """메인 실행 함수"""
    system = None

    try:
        system = TeleopSystem()

        if not system.initialized:
            print("\n❌ System initialization failed")
            print("\nTroubleshooting:")
            print("  1. Install required packages: pip install opencv-python numpy mediapipe")
            print("  2. Check camera connection")
            print("  3. Verify all Python files are in correct locations")
            return

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
