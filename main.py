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


class TeleopApplication:
    """메인 텔레오퍼레이션 애플리케이션"""

    def __init__(self):
        # 하드웨어 및 비전 초기화
        self.gripper = DG5FDevClient(GRIPPER_IP, GRIPPER_PORT, timeout=0.5)
        self.hand_tracker = HandTrackerTasks(MODEL_PATH)
        self.motor_controller = MotorController()
        self.hud_manager = HUDManager()

        # 카메라
        self.camera = None

        # 화면 표시 관련 변수
        self.window_name = "DG-5F Dev Teleop"
        self.display_scale = 1.0  # 화면 표시 배율

        # 애플리케이션 상태 (복구된 변수들!)
        self.emergency_stop = False
        self.home_zero = False

        # 사용자 입력 상태 (복구된 변수들!)
        self.motor_input_buffer = ""
        self.flash_text = ""
        self.flash_end_time = 0.0  # ← 에러의 직접적 원인

        # 마지막 전송된 상태
        self.last_sent_duty = make_zero_duty()
        self.last_target = {m: 0 for m in range(1, 21)}

        # F/T 센서 관련 변수 (추가!)
        self.ft_data = None                # 최근 F/T 센서 데이터
        self.show_ft_data = False          # F/T 데이터 화면 표시 여부
        self.last_ft_read_time = 0.0       # 마지막 F/T 읽기 시간
        self.ft_read_interval = 0.05       # F/T 읽기 주기 (20Hz)


    def initialize(self) -> bool:
        """하드웨어 연결 초기화"""
        logger.info("=" * 55)
        logger.info("DG-5F-M Developer Mode Teleoperation")
        logger.info("Key Controls:")
        logger.info("  q: Quit | z: Zero Duty | x: Emergency Stop | r: Home Zero")
        logger.info("  Number Input:")
        logger.info("    0-20 + Enter: Individual Motor ON/OFF")
        logger.info("    1-5 + F: Finger Unit ON/OFF (1:Thumb 2:Index 3:Middle 4:Ring 5:Pinky)")
        logger.info("  T: Toggle ALL Motors | Backspace: Clear Input")
        logger.info("  +/-: Resize Window")
        logger.info("=" * 55)

        # 그리퍼 연결
        if not self.gripper.connect():
            logger.error("그리퍼 연결 실패")
            return False

        # 카메라 초기화
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("카메라 열기 실패")
            self.gripper.close()
            return False

        # 카메라 해상도 설정 (더 큰 화면)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 기본 640 → 1280
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 기본 480 → 720

        # 실제 적용된 해상도 확인 및 로그
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"카메라 해상도: {actual_width}x{actual_height}")

        # OpenCV 윈도우 생성 및 크기 설정 (추가!)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 크기 조절 가능
        initial_width = int(actual_width * 1.2)  # 카메라 해상도의 1.2배로 시작
        initial_height = int(actual_height * 1.2)
        cv2.resizeWindow(self.window_name, initial_width, initial_height)
        logger.info(f"초기 윈도우 크기: {initial_width}x{initial_height}")

        logger.info("초기화 완료")
        return True

    def cleanup(self):
        """리소스 정리"""
        try:
            # 안전을 위해 모든 듀티를 0으로 설정
            self.gripper.set_duty(make_zero_duty())
            logger.info("안전 정지 명령 전송됨")
        except Exception as e:
            logger.warning(f"안전 정지 실패: {e}")

        # 연결 종료
        self.gripper.close()

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        logger.info("리소스 정리 완료")

    def handle_keyboard_input(self, key: int, current_positions: Dict[int, int]) -> bool:
        """
        키보드 입력 처리

        Returns: True if should quit
        """
        # 플래시 텍스트 타임아웃 확인
        if time.time() > self.flash_end_time:
            self.flash_text = ""

        # 종료
        if key == ord("q"):
            return True

        # 숫자 입력
        if ord('0') <= key <= ord('9'):
            if len(self.motor_input_buffer) < 2:
                self.motor_input_buffer += chr(key)

        # 백스페이스
        if key in (8, 127):  # Backspace, Delete
            self.motor_input_buffer = self.motor_input_buffer[:-1]

        # 엔터: 모터 토글
        if key in (10, 13):  # Enter, Return
            if self.motor_input_buffer:
                try:
                    motor_id = int(self.motor_input_buffer)
                    message = self.motor_controller.toggle_motor_enable(motor_id, current_positions)
                    self.flash_text = message
                    self.flash_end_time = time.time() + 1.5
                    self.motor_input_buffer = ""
                except ValueError:
                    self.flash_text = "Invalid motor ID"  # 변경!
                    self.flash_end_time = time.time() + 1.5

        # T: 모든 모터 ON/OFF 토글
        if key == ord("t"):
            # 현재 모든 모터가 비활성화되어 있는지 확인
            all_disabled = all(not enabled for enabled in self.motor_controller.motor_enabled.values())

            if all_disabled:
                # 모두 비활성화 상태 → 모두 활성화
                for motor_id in range(1, 21):
                    self.motor_controller.motor_enabled[motor_id] = True
                self.flash_text = "All motors ENABLED (20/20 ON)"  # 변경!
                logger.info("모든 모터 활성화")
            else:
                # 일부 또는 전체 활성화 상태 → 모두 비활성화
                for motor_id in range(1, 21):
                    self.motor_controller.motor_enabled[motor_id] = False
                    # ... 기존 로직 ...
                self.flash_text = "All motors DISABLED (0/20 ON)"  # 변경!
                logger.info("모든 모터 비활성화")

            # 격리 모드 완전 해제 (기존 코드와의 호환성)
            self.motor_controller.isolate_mode = False
            self.motor_controller.isolate_motor_id = None

            # 입력 버퍼 초기화
            self.motor_input_buffer = ""
            self.flash_end_time = time.time() + 1.5

        # F: 손가락 단위 모터 ON/OFF 토글 (새로 추가!)
        if key == ord("f"):
            if self.motor_input_buffer:
                try:
                    finger_num = int(self.motor_input_buffer)

                    if not (1 <= finger_num <= 5):
                        self.flash_text = "Finger number must be 1-5"  # 변경!
                        self.flash_end_time = time.time() + 1.5
                        self.motor_input_buffer = ""
                        return False

                    # 손가락별 모터 매핑 (수학적 계산)
                    start_motor = (finger_num - 1) * 4 + 1
                    motor_ids = list(range(start_motor, start_motor + 4))

                    # 손가락 이름 매핑 (영문으로 변경)
                    finger_names = {
                        1: "Thumb",
                        2: "Index",
                        3: "Middle",
                        4: "Ring",
                        5: "Pinky"
                    }

                    # 해당 손가락의 모터들이 하나라도 켜져 있는지 확인
                    any_enabled = any(self.motor_controller.motor_enabled[mid] for mid in motor_ids)

                    if any_enabled:
                        # 하나라도 켜져 있으면 → 모두 끄기
                        for motor_id in motor_ids:
                            self.motor_controller.motor_enabled[motor_id] = False
                            # 현재 위치로 안전하게 고정
                            hold_position = int(current_positions.get(motor_id,
                                                                      self.motor_controller.previous_target.get(
                                                                          motor_id, 0)))
                            self.motor_controller.previous_target[motor_id] = hold_position
                            self.motor_controller.previous_duty[motor_id] = 0

                        self.flash_text = f"{finger_names[finger_num]} DISABLED (M{motor_ids[0]}-{motor_ids[-1]})"  # 변경!
                        logger.info(f"{finger_names[finger_num]} 모터 비활성화: {motor_ids}")
                    else:
                        # 모두 꺼져 있으면 → 모두 켜기
                        for motor_id in motor_ids:
                            self.motor_controller.motor_enabled[motor_id] = True

                        self.flash_text = f"{finger_names[finger_num]} ENABLED (M{motor_ids[0]}-{motor_ids[-1]})"  # 변경!
                        logger.info(f"{finger_names[finger_num]} 모터 활성화: {motor_ids}")

                    # 격리 모드 해제 (호환성 보장)
                    self.motor_controller.isolate_mode = False
                    self.motor_controller.isolate_motor_id = None

                    # 입력 버퍼 초기화
                    self.motor_input_buffer = ""
                    self.flash_end_time = time.time() + 1.5


                except ValueError:
                    self.flash_text = "Invalid finger number"  # 변경!
                    self.flash_end_time = time.time() + 1.5
                    self.motor_input_buffer = ""

                except Exception as e:
                    self.flash_text = f"Finger toggle error: {str(e)}"  # 변경!
                    self.flash_end_time = time.time() + 1.5
                    logger.error(f"손가락 토글 처리 실패: {e}")
                    self.motor_input_buffer = ""
                else:
                    self.flash_text = "Enter finger number first (1-5)"  # 변경!
                    self.flash_end_time = time.time() + 1.5

        # Z: 듀티 제로
        if key == ord("z"):
            try:
                self.gripper.set_duty(make_zero_duty())
                self.motor_controller.reset_duty_state()
                self.last_sent_duty = make_zero_duty()
                logger.info("모든 듀티 제로로 설정")
            except Exception as e:
                logger.error(f"듀티 제로 설정 실패: {e}")

        # X: 비상 정지
        if key == ord("x"):
            self.emergency_stop = not self.emergency_stop
            if self.emergency_stop:
                self.home_zero = False
            logger.info(f"비상정지 = {self.emergency_stop}")

        # R: 홈 제로
        if key == ord("r"):
            self.home_zero = not self.home_zero
            if self.home_zero:
                self.emergency_stop = False
            logger.info(f"홈 제로 = {self.home_zero}")

        # '+' 또는 '=' 키: 화면 확대
        if key == ord("+") or key == ord("="):
            self.display_scale = min(3.0, self.display_scale * 1.1)
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) * self.display_scale)
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.display_scale)
            cv2.resizeWindow(self.window_name, width, height)
            self.flash_text = f"Scale: {self.display_scale:.1f}x"  # 변경!
            self.flash_end_time = time.time() + 1.0

        # '-' 키: 화면 축소
        if key == ord("-"):
            self.display_scale = max(0.5, self.display_scale / 1.1)
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) * self.display_scale)
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.display_scale)
            cv2.resizeWindow(self.window_name, width, height)
            self.flash_text = f"Scale: {self.display_scale:.1f}x"  # 변경!
            self.flash_end_time = time.time() + 1.0


        return False


    def handle_emergency_stop(self, frame: np.ndarray, current_positions: Dict[int, int],
                              landmarks_xy: Optional[list]):
        """비상 정지 상태 처리"""
        try:
            self.gripper.set_duty(make_zero_duty())
            self.motor_controller.reset_targets_to_current(current_positions)
            self.motor_controller.reset_duty_state()
            self.last_sent_duty = make_zero_duty()
        except Exception as e:
            logger.error(f"비상 정지 처리 실패: {e}")

        # HUD 업데이트
        self.hud_manager.render_frame(
            frame,
            status_text="EMERGENCY STOP ACTIVE",
            landmarks_xy=landmarks_xy,
            current_positions=current_positions,
            target_positions=self.last_target,
            duty_dict=self.last_sent_duty,
            emergency_stop=self.emergency_stop,
            home_zero=self.home_zero,
            motor_input_buffer=self.motor_input_buffer,
            flash_text=self.flash_text,
            isolate_mode=self.motor_controller.isolate_mode,
            isolate_motor_id=self.motor_controller.isolate_motor_id,
            motor_enabled=self.motor_controller.motor_enabled
        )

        cv2.imshow(self.window_name, frame)

    def handle_home_zero_mode(self, frame: np.ndarray, current_positions: Dict[int, int],
                              landmarks_xy: Optional[list]):
        """홈 제로 모드 처리"""
        try:
            # 모든 모터를 Base Angle로 목표 설정 (실제 0도 위치)
            desired = {
                m: int(BASE_ANGLES.get(m, 0.0) * 10)  # 도 → 0.1도 단위
                for m in range(1, 21)
            }


            # 제어 파이프라인 적용
            desired = self.motor_controller.apply_step_limits(desired, current_positions)

            # 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, desired=desired)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, desired=desired)

            # 속도 제한
            target = self.motor_controller.apply_speed_limits(desired)
            self.last_target = dict(target)

            # 마스크 재적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, target=target)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, target=target)

            # 원시 듀티 계산
            raw_duty = self.motor_controller.compute_raw_duty(target, current_positions)

            # 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, raw=raw_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, raw=raw_duty)

            # 전역 제한 적용
            raw_duty = apply_global_limits(raw_duty)

            # 마스크 재적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, raw=raw_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, raw=raw_duty)

            # 슬루 제한 적용
            final_duty = self.motor_controller.apply_duty_slew(raw_duty)

            # 최종 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, duty=final_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, duty=final_duty)

            # 그리퍼에 전송
            self.gripper.set_duty(final_duty)
            self.last_sent_duty = dict(final_duty)

        except Exception as e:
            logger.error(f"홈 제로 모드 처리 실패: {e}")


        # 상태 텍스트 생성
        status = f"HOME0 Mode (Budget={GLOBAL_LIMITS['total_duty_budget']} TopK={GLOBAL_LIMITS['max_active_joints']})"  # 변경!
        if self.motor_controller.isolate_mode and self.motor_controller.isolate_motor_id:
            status += f" | Isolate M{self.motor_controller.isolate_motor_id:02d}"


        # HUD 업데이트
        self.hud_manager.render_frame(
            frame,
            status_text=status,
            landmarks_xy=landmarks_xy,
            current_positions=current_positions,
            target_positions=self.last_target,
            duty_dict=self.last_sent_duty,
            emergency_stop=self.emergency_stop,
            home_zero=self.home_zero,
            motor_input_buffer=self.motor_input_buffer,
            flash_text=self.flash_text,
            isolate_mode=self.motor_controller.isolate_mode,
            isolate_motor_id=self.motor_controller.isolate_motor_id,
            motor_enabled=self.motor_controller.motor_enabled
        )

        cv2.imshow(self.window_name, frame)

    def handle_teleoperation_mode(self, frame: np.ndarray, current_positions: Dict[int, int],
                                  curls: Dict, splay: Dict, thumb_features: tuple,
                                  landmarks_xy: Optional[list]):
        """일반 텔레오퍼레이션 모드 처리"""
        try:
            # 스무딩 적용
            curls, splay, thumb_features = self.motor_controller.apply_smoothing(
                curls, splay, thumb_features)

            thumb_mcp_curl, thumb_ip_curl, thumb_opposition = thumb_features

            # 원하는 목표값 계산
            desired = self.motor_controller.compute_desired_targets(
                curls, splay, thumb_mcp_curl, thumb_ip_curl, thumb_opposition)

            # 제어 파이프라인 적용
            desired = self.motor_controller.apply_step_limits(desired, current_positions)

            # 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, desired=desired)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, desired=desired)

            # 속도 제한
            target = self.motor_controller.apply_speed_limits(desired)
            self.last_target = dict(target)

            # 마스크 재적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, target=target)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, target=target)

            # 원시 듀티 계산
            raw_duty = self.motor_controller.compute_raw_duty(target, current_positions)

            # 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, raw=raw_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, raw=raw_duty)

            # 전역 제한 적용
            raw_duty = apply_global_limits(raw_duty)

            # 마스크 재적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, raw=raw_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, raw=raw_duty)

            # 슬루 제한 적용
            final_duty = self.motor_controller.apply_duty_slew(raw_duty)

            # 최종 마스크 적용
            self.motor_controller.enforce_motor_enable_mask(
                current_positions, duty=final_duty)
            if self.motor_controller.isolate_mode:
                self.motor_controller.enforce_isolate_mode(
                    current_positions, duty=final_duty)

            # 그리퍼에 전송
            write_error = None
            try:
                self.gripper.set_duty(final_duty)
                self.last_sent_duty = dict(final_duty)
            except Exception as e:
                write_error = f"쓰기 오류: {e}"
                logger.error(write_error)

                # 오류 시 안전 조치
                try:
                    self.gripper.set_duty(make_zero_duty())
                except:
                    pass
                self.motor_controller.reset_targets_to_current(current_positions)
                self.motor_controller.reset_duty_state()
                self.last_sent_duty = make_zero_duty()

        except Exception as e:
            logger.error(f"텔레오퍼레이션 모드 처리 실패: {e}")
            write_error = f"처리 오류: {e}"

        # 상태 텍스트 생성
        status = f"Teleoperation | Budget={GLOBAL_LIMITS['total_duty_budget']} TopK={GLOBAL_LIMITS['max_active_joints']}"  # 변경!
        if self.motor_controller.isolate_mode and self.motor_controller.isolate_motor_id:
            status += f" | Isolate M{self.motor_controller.isolate_motor_id:02d}"
        if write_error:
            status = write_error

        # HUD 업데이트
        self.hud_manager.render_frame(
            frame,
            status_text=status,
            landmarks_xy=landmarks_xy,
            current_positions=current_positions,
            target_positions=self.last_target,
            duty_dict=self.last_sent_duty,
            emergency_stop=self.emergency_stop,
            home_zero=self.home_zero,
            motor_input_buffer=self.motor_input_buffer,
            flash_text=self.flash_text,
            isolate_mode=self.motor_controller.isolate_mode,
            isolate_motor_id=self.motor_controller.isolate_motor_id,
            motor_enabled=self.motor_controller.motor_enabled
        )

        cv2.imshow(self.window_name, frame)

    def run(self):
        """메인 애플리케이션 루프"""
        last_time = time.time()

        try:
            while True:
                # 타이밍 제어
                current_time = time.time()
                if current_time - last_time < DT:
                    time.sleep(max(0.0, DT - (current_time - last_time)))
                last_time = time.time()

                # 프레임 캡처
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("프레임 캡처 실패")
                    break

                # 프레임 좌우 반전 (거울 효과)
                frame = cv2.flip(frame, 1)

                # 핸드 트래킹 처리
                frame, curls, splay, thumb_features, landmarks_xy = self.hand_tracker.process(frame)

                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF

                # 현재 위치 획득 및 Base Angle 보정 적용
                try:
                    raw_positions = self.gripper.get_positions()
                    current_positions = self.motor_controller.apply_base_angle_correction(raw_positions)
                except Exception as e:
                    logger.error(f"위치 읽기 실패: {e}")

                    # 통신 오류 시 안전 조치
                    try:
                        self.gripper.set_duty(make_zero_duty())
                    except:
                        pass

                    self.motor_controller.reset_duty_state()
                    self.last_sent_duty = make_zero_duty()

                    # 오류 상태 HUD 표시
                    self.hud_manager.render_frame(
                        frame,
                        status_text=f"Communication Error: {e}",  # 변경!
                        landmarks_xy=landmarks_xy,
                        current_positions=None,
                        target_positions=None,
                        duty_dict=self.last_sent_duty,
                        emergency_stop=self.emergency_stop,
                        home_zero=self.home_zero,
                        motor_input_buffer=self.motor_input_buffer,
                        flash_text=self.flash_text,
                        isolate_mode=self.motor_controller.isolate_mode,
                        isolate_motor_id=self.motor_controller.isolate_motor_id,
                        motor_enabled=self.motor_controller.motor_enabled
                    )
                    cv2.imshow(self.window_name, frame)

                    self.motor_controller.target_valid = False
                    continue

                # 키보드 입력 처리
                if self.handle_keyboard_input(key, current_positions):
                    break

                # 목표값 초기화 (필요시)
                if not self.motor_controller.target_valid:
                    self.motor_controller.reset_targets_to_current(current_positions)
                    self.motor_controller.reset_duty_state()

                # 모드별 처리
                if self.emergency_stop:
                    self.handle_emergency_stop(frame, current_positions, landmarks_xy)
                    continue

                if self.home_zero:
                    self.handle_home_zero_mode(frame, current_positions, landmarks_xy)
                    continue

                # 손이 감지되지 않음
                if curls is None or splay is None or thumb_features is None:
                    try:
                        self.gripper.set_duty(make_zero_duty())
                        self.motor_controller.reset_targets_to_current(current_positions)
                        self.motor_controller.reset_duty_state()
                        self.last_sent_duty = make_zero_duty()
                    except Exception as e:
                        logger.error(f"손 미감지 시 안전 조치 실패: {e}")

                    self.hud_manager.render_frame(
                        frame,
                        status_text="Show your hand",  # 변경!
                        landmarks_xy=landmarks_xy,
                        current_positions=current_positions,
                        target_positions=self.last_target,
                        duty_dict=self.last_sent_duty,
                        emergency_stop=self.emergency_stop,
                        home_zero=self.home_zero,
                        motor_input_buffer=self.motor_input_buffer,
                        flash_text=self.flash_text,
                        isolate_mode=self.motor_controller.isolate_mode,
                        isolate_motor_id=self.motor_controller.isolate_motor_id,
                        motor_enabled=self.motor_controller.motor_enabled
                    )
                    cv2.imshow(self.window_name, frame)
                    continue

                # 일반 텔레오퍼레이션
                self.handle_teleoperation_mode(
                    frame, current_positions, curls, splay, thumb_features, landmarks_xy)

        except KeyboardInterrupt:
            logger.info("사용자에 의한 중단")
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
        finally:
            self.cleanup()


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
        app = TeleopApplication()

        if not app.initialize():
            logger.error("초기화 실패")
            return 1

        app.run()
        return 0

    except Exception as e:
        logger.error(f"애플리케이션 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    system = TeleopSystem()

    try:
        if system.initialize_hardware():
            system.main_control_loop()
        else:
            print("❌ Failed to initialize hardware")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        system.shutdown()

    sys.exit(main())
