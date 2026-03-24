#!/usr/bin/env python3
#main.py
"""
DG-5F-M 개발자 모드 텔레오퍼레이션
메인 애플리케이션 진입점
"""
import time
import cv2
import logging
import sys
import numpy as np
from typing import Dict, Optional
from ui.ft_dashboard import FTDashboard
from hardware.ft_sensor_client import FTSensorClient  # 또는 실제 클래스명
from ui.ft_dashboard import FTDashboard


# 로컬 모듈 임포트
from config.constants import (
    GRIPPER_IP, GRIPPER_PORT, MODEL_PATH,
    CONTROL_HZ, DT, GLOBAL_LIMITS, BASE_ANGLES
)
from hardware.gripper_client import DG5FDevClient
from vision.hand_tracker import HandTrackerTasks
from control.motor_controller import MotorController
from control.safety import make_zero_duty, apply_global_limits
from ui.visualization import HUDManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeleopSystem:
    def __init__(self):
        self.running = True
        self.emergency_stop = False

        # 카메라 초기화
        self.camera = cv2.VideoCapture(0)

        # F/T 센서 및 대시보드 초기화 (안전한 방식)
        try:
            self.ft_sensor = FTSensorClient()
            self.ft_dashboard = FTDashboard(width=1000, height=700)
            self.ft_sensor_enabled = False
            print("✓ F/T 시스템 객체 생성됨")
        except Exception as e:
            print(f"⚠ F/T 시스템 초기화 경고: {e}")
            self.ft_sensor = None
            self.ft_dashboard = None
            self.ft_sensor_enabled = False

    def initialize_hardware(self):
        """하드웨어 초기화"""
        print("하드웨어 초기화 중...")

        # F/T 센서 연결 시도
        if self.ft_sensor:
            try:
                if self.ft_sensor.connect():
                    if self.ft_sensor.start_reading():
                        print("✓ F/T 센서 연결 및 읽기 시작")
                        time.sleep(1.0)  # 안정화 대기
                        self.ft_sensor.set_bias()  # 영점 설정
                        self.ft_sensor_enabled = True
                    else:
                        print("⚠ F/T 센서 연결됨, 하지만 읽기 실패")
                else:
                    print("⚠ F/T 센서 연결 실패 (시뮬레이션 모드)")
            except Exception as e:
                print(f"⚠ F/T 센서 에러: {e}")

        return True

    def main_control_loop(self):
        """메인 제어 루프"""
        print("제어 루프 시작...")
        print("조작법: Q-종료, Z-센서영점재설정, R-비상정지해제")

        while self.running:
            # 카메라 프레임 읽기
            ret, frame = self.camera.read()
            if not ret:
                continue

            # F/T 센서 데이터 처리
            if self.ft_sensor_enabled and self.ft_sensor:
                try:
                    # 센서 데이터 가져오기
                    force, torque = self.ft_sensor.get_force_torque(filtered=True)

                    # 대시보드 데이터 업데이트
                    if self.ft_dashboard:
                        self.ft_dashboard.update_data(force, torque)

                    # 안전 체크
                    exceeded, safety_msg = self.ft_sensor.check_safety_limits()
                    if exceeded and not self.emergency_stop:
                        print(f"🚨 {safety_msg}")
                        self.emergency_stop = True

                except Exception as e:
                    print(f"⚠ F/T 처리 에러: {e}")

            # 화면 표시 (핵심: 모든 imshow를 메인 스레드에서 처리)
            try:
                # 메인 카메라 화면
                self._draw_main_info(frame)
                cv2.imshow("Tesollo Hand Teleoperation", frame)

                # F/T 대시보드 화면 (별도 창)
                if self.ft_dashboard:
                    dashboard_frame = self.ft_dashboard.get_dashboard_frame()
                    cv2.imshow("F/T Sensor Dashboard", dashboard_frame)

            except Exception as e:
                print(f"⚠ 화면 표시 에러: {e}")

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z') and self.ft_sensor_enabled:
                self.ft_sensor.set_bias()
                print("✓ F/T 센서 영점 재설정")
            elif key == ord('r'):
                if self.emergency_stop:
                    self.emergency_stop = False
                    print("✓ 비상 정지 해제")

    def _draw_main_info(self, frame):
        """메인 화면에 상태 정보 표시"""
        h, w = frame.shape[:2]

        # 상태 바 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # F/T 센서 상태
        if self.ft_sensor_enabled:
            status_color = (0, 0, 255) if self.emergency_stop else (0, 255, 0)
            status_text = "E-STOP" if self.emergency_stop else "ACTIVE"
            cv2.putText(frame, f"F/T: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(frame, "F/T: OFFLINE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    def shutdown(self):
        """시스템 종료"""
        print("시스템 종료 중...")

        if self.ft_sensor_enabled and self.ft_sensor:
            self.ft_sensor.disconnect()

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        print("✓ 종료 완료")


class TeleopSystem:
    def __init__(self):
        # ... 기존 초기화 ...

        # F/T 대시보드 추가
        self.ft_dashboard = FTDashboard(
            width=1000,
            height=700,
            history_length=300  # 10초 히스토리
        )

    def initialize_hardware(self) -> bool:
        """하드웨어 초기화"""
        print("Initializing hardware components...")

        # 1. 기존 하드웨어 초기화 (gripper, camera 등)
        # ... 기존 코드 ...

        # 2. F/T 센서 초기화
        if self.ft_sensor.connect():
            if self.ft_sensor.start_reading():
                time.sleep(1.0)  # 센서 안정화 대기
                self.ft_sensor.set_bias()  # 영점 설정
                self.ft_sensor_enabled = True

                # F/T 대시보드 시작
                self.ft_dashboard.start()
                print("✓ F/T Sensor and Dashboard initialized")
            else:
                print("⚠ F/T Sensor connected but failed to start reading")
        else:
            print("⚠ F/T Sensor not available")
            self.ft_sensor_enabled = False

        return True

    def main_control_loop(self):
        """메인 제어 루프"""
        print("Starting teleoperation with F/T monitoring...")

        while self.running:
            loop_start = time.time()

            # 1. 카메라 프레임 읽기
            frame = self.camera.read()
            if frame is None:
                continue

            # 2. 손 추적 처리
            hand_data = self.hand_tracker.process(frame)

            # 3. F/T 센서 데이터 처리
            ft_safe = True
            if self.ft_sensor_enabled:
                # 실시간 F/T 데이터 가져오기
                force, torque = self.ft_sensor.get_force_torque(filtered=True)

                # 🎯 대시보드 업데이트 (핵심!)
                self.ft_dashboard.update_data(force, torque)

                # 안전 체크
                exceeded, safety_msg = self.ft_sensor.check_safety_limits()
                if exceeded:
                    print(f"⚠ {safety_msg}")
                    ft_safe = False

                    # 비상 정지
                    if self.ft_dashboard.get_dashboard_status()['emergency_state']:
                        self.emergency_stop = True
                        self.gripper.stop_all_motors()
                        print("🚨 EMERGENCY STOP activated by F/T limits!")

            # 4. 모터 제어 (안전할 때만)
            if hand_data and ft_safe and not self.emergency_stop:
                motor_commands = self.motor_controller.compute_commands(hand_data)
                safe_commands = self.safety.validate_commands(motor_commands)
                self.gripper.send_commands(safe_commands)

            # 5. 메인 화면 표시 (기존 카메라 창)
            self.visualizer.update_main_display(frame, hand_data, self.emergency_stop)
            cv2.imshow("Tesollo Hand Teleoperation", frame)

            # 6. 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z') and self.ft_sensor_enabled:
                # F/T 센서 영점 재설정
                self.ft_sensor.set_bias()
                print("✓ F/T Sensor re-biased")
            elif key == ord('r'):
                # 비상 정지 해제 및 통계 리셋
                if self.emergency_stop:
                    self.emergency_stop = False
                    print("✓ Emergency stop reset")
                if self.ft_sensor_enabled:
                    self.ft_dashboard._reset_statistics()
            elif key == ord('s'):
                # 시스템 상태 출력
                self._print_system_status()

            # 7. 루프 타이밍 (30fps 유지)
            loop_time = time.time() - loop_start
            if loop_time < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - loop_time)

    def _print_system_status(self):
        """상세 시스템 상태 출력"""
        print("\n" + "=" * 50)
        print("TESOLLO TELEOPERATION SYSTEM STATUS")
        print("=" * 50)

        # F/T 대시보드 상태
        if self.ft_sensor_enabled:
            dashboard_status = self.ft_dashboard.get_dashboard_status()
            force_mag, torque_mag = self.ft_sensor.get_magnitude()

            print(f"F/T Sensor: ✓ Active")
            print(f"  Current Force: {force_mag:.2f}N (Max: {MAX_FORCE_LIMIT}N)")
            print(f"  Current Torque: {torque_mag:.3f}Nm (Max: {MAX_TORQUE_LIMIT}Nm)")
            print(f"  Dashboard: {'✓ Running' if dashboard_status['running'] else '✗ Stopped'}")
            print(f"  Emergency State: {'🚨 ACTIVE' if dashboard_status['emergency_state'] else '✓ Normal'}")
            print(f"  Max Recorded Force: {dashboard_status['max_force_recorded']:.2f}N")
            print(f"  Max Recorded Torque: {dashboard_status['max_torque_recorded']:.3f}Nm")
        else:
            print(f"F/T Sensor: ✗ Not Available")

        # 그리퍼 상태
        print(f"Gripper: {'✓ Connected' if self.gripper.connected else '✗ Disconnected'}")

        # 전체 시스템 상태
        print(f"Emergency Stop: {'🚨 ACTIVE' if self.emergency_stop else '✓ Inactive'}")
        print("=" * 50 + "\n")

    def shutdown(self):
        """시스템 종료"""
        print("Shutting down Tesollo Teleoperation System...")

        # F/T 대시보드 종료
        if hasattr(self, 'ft_dashboard'):
            self.ft_dashboard.stop()

        # F/T 센서 종료
        if self.ft_sensor_enabled:
            self.ft_sensor.disconnect()

        # 기존 하드웨어 종료
        if hasattr(self, 'gripper'):
            self.gripper.disconnect()

        # 모든 OpenCV 창 닫기
        cv2.destroyAllWindows()

        print("✓ System shutdown complete")


def main():
    """애플리케이션 진입점"""
    try:
        app = TeleopSystem()

        if not app.initialize():
            logger.error("초기화 실패")
            return 1

        app.run()
        return 0

    except Exception as e:
        logger.error(f"애플리케이션 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    system = None
    try:
        system = TeleopSystem()
        if system.initialize_hardware():
            system.main_control_loop()
        else:
            print("❌ 하드웨어 초기화 실패")
    except KeyboardInterrupt:
        print("\n사용자에 의한 중단")
    except Exception as e:
        print(f"❌ 시스템 에러: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.shutdown()
