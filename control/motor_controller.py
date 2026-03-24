#motor_controller.py
"""
모터 제어 로직 및 상태 관리
"""
import numpy as np
import logging
from typing import Dict, Tuple, Optional

from config.constants import *

logger = logging.getLogger(__name__)


class MotorController:
    """모터 제어 상태 및 계산 관리 클래스"""

    def __init__(self):
        self.motor_enabled = {m: True for m in range(1, 21)}
        self.previous_target = {m: 0 for m in range(1, 21)}
        self.previous_duty = {m: 0 for m in range(1, 21)}
        self.target_valid = False

        # 스무딩 상태
        self.smooth_curl = {f: 0.0 for f in FINGER_ORDER}
        self.smooth_splay = {f: 0.0 for f in FINGER_ORDER}
        self.smooth_thumb_mcp = 0.0
        self.smooth_thumb_ip = 0.0
        self.smooth_thumb_opposition = 0.0  # ← 추가!


        # 격리 모드
        self.isolate_mode = False
        self.isolate_motor_id = None




    def reset_targets_to_current(self, current_positions: Dict[int, int]):
        """목표 위치를 현재 위치로 재설정"""
        for motor_id in range(1, 21):
            value = int(current_positions.get(motor_id, 0))
            self.previous_target[motor_id] = value
        self.target_valid = True


    def reset_duty_state(self):
        """듀티 사이클 상태 재설정"""
        for motor_id in range(1, 21):
            self.previous_duty[motor_id] = 0


    def toggle_motor_enable(self, motor_id: int, current_positions: Dict[int, int]) -> str:
        """모터 활성화/비활성화 상태 토글"""
        if not (1 <= motor_id <= 20):
            return f"유효하지 않은 모터 ID: {motor_id}"

        self.motor_enabled[motor_id] = not self.motor_enabled[motor_id]

        # 현재 위치로 홀드 설정
        hold_position = int(current_positions.get(motor_id, self.previous_target.get(motor_id, 0)))
        self.previous_target[motor_id] = hold_position
        self.previous_duty[motor_id] = 0

        status = "ENABLED" if self.motor_enabled[motor_id] else "DISABLED"  # 변경!
        message = f"Motor {motor_id:02d} ({MOTOR_ROLES[motor_id]}) -> {status}"
        logger.info(message)
        return message

    def enter_isolate_mode(self, motor_id: int, current_positions: Dict[int, int]) -> str:
        """단일 모터 격리 모드 진입"""
        if not (1 <= motor_id <= 20):
            return f"격리 모드에 유효하지 않은 모터 ID: {motor_id}"

        self.isolate_mode = True
        self.isolate_motor_id = motor_id

        # 모든 모터를 현재 위치로 설정
        for m in range(1, 21):
            self.previous_target[m] = int(current_positions.get(m, 0))
            self.previous_duty[m] = 0

        message = f"격리 모드 활성화 - M{motor_id:02d} ({MOTOR_ROLES[motor_id]})"
        logger.info(message)
        return message

    def exit_isolate_mode(self, current_positions: Dict[int, int]) -> str:
        """격리 모드 종료"""
        for m in range(1, 21):
            self.previous_target[m] = int(current_positions.get(m, 0))
            self.previous_duty[m] = 0

        self.isolate_mode = False
        self.isolate_motor_id = None

        message = "격리 모드 비활성화"
        logger.info(message)
        return message

    def apply_smoothing(self, curls: Dict, splay: Dict, thumb_features: Tuple) -> Tuple:
        """핸드 트래킹 특징에 스무딩 적용"""
        # 컬 값 스무딩
        for finger in FINGER_ORDER:
            self.smooth_curl[finger] = ((1.0 - SMOOTH_ALPHA) * self.smooth_curl[finger] +
                                        SMOOTH_ALPHA * curls[finger])
            curls[finger] = self.smooth_curl[finger]

            self.smooth_splay[finger] = ((1.0 - SPLAY_SMOOTH_ALPHA) * self.smooth_splay[finger] +
                                         SPLAY_SMOOTH_ALPHA * splay[finger])
            splay[finger] = self.smooth_splay[finger]

        # 엄지 특징 스무딩 (3개 값으로 확장)
        thumb_mcp_curl, thumb_ip_curl, thumb_opposition = thumb_features

        self.smooth_thumb_mcp = ((1.0 - SMOOTH_ALPHA) * self.smooth_thumb_mcp +
                                 SMOOTH_ALPHA * thumb_mcp_curl)
        self.smooth_thumb_ip = ((1.0 - SMOOTH_ALPHA) * self.smooth_thumb_ip +
                                SMOOTH_ALPHA * thumb_ip_curl)
        self.smooth_thumb_opposition = ((1.0 - SMOOTH_ALPHA) * self.smooth_thumb_opposition +
                                        SMOOTH_ALPHA * thumb_opposition)

        return (curls, splay,
                (self.smooth_thumb_mcp, self.smooth_thumb_ip, self.smooth_thumb_opposition))

    def compute_desired_targets(self, curls: Dict, splay: Dict,
                                thumb_mcp_curl: float, thumb_ip_curl: float,
                                thumb_opposition: float) -> Dict[int, int]:
        """손 특징으로부터 원하는 모터 목표값 계산"""
        desired = {m: self.previous_target[m] for m in range(1, 21)}

        for finger in FINGER_ORDER:
            j0, j1, j2, j3 = JOINT_MAP[finger]

            if finger == "finger1":  # 엄지 특별 처리
                # Motor 1: 벌림(Spread) - 기존 splay 사용
                spread_deg = float(np.clip(
                    MOTION_RANGES['splay_thumb_gain'] * splay[finger],
                    -MOTION_RANGES['splay_thumb_limit'],
                    MOTION_RANGES['splay_thumb_limit']
                ))

                # Motor 2: 대립(Opposition) - 새로운 계산 ← 핵심!
                # 0.0(펼침) -> 0도, 1.0(대립) -> -150도
                opposition_deg = -thumb_opposition * MOTION_RANGES['opposition_thumb_limit']

                # Motor 3, 4: MCP, IP 굽힘
                mcp_deg = self._curl_to_flex_deg(thumb_mcp_curl, MOTION_RANGES['flex_thumb_mcp'])
                ip_deg = self._curl_to_flex_deg(thumb_ip_curl, MOTION_RANGES['flex_thumb_ip'])

                desired[j0] = self._clamp_target(j0, TARGET_SIGN[j0] * int(spread_deg * 10))  # Motor 1
                desired[j1] = self._clamp_target(j1, TARGET_SIGN[j1] * int(opposition_deg * 10))  # Motor 2
                desired[j2] = self._clamp_target(j2, TARGET_SIGN[j2] * int(mcp_deg * 10))  # Motor 3
                desired[j3] = self._clamp_target(j3, TARGET_SIGN[j3] * int(ip_deg * 10))  # Motor 4

            elif finger == "finger5":  # 새끼손가락 특별 처리
                # Motor 17, 18: 벌림 계열 (spread/adduction)
                # Motor 19, 20: 굽힘 계열 (flexion)
                spread_deg = float(np.clip(
                    MOTION_RANGES['splay_default_gain'] * splay[finger],
                    -MOTION_RANGES['splay_default_limit'],
                    MOTION_RANGES['splay_default_limit']
                ))
                flex_deg = self._curl_to_flex_deg(curls[finger], MOTION_RANGES['flex_default'])
                spread_command = int(spread_deg * 10)
                flex_command = int(flex_deg * 10)
                desired[j0] = self._clamp_target(j0, TARGET_SIGN[j0] * spread_command)  # Motor 17
                desired[j1] = self._clamp_target(j1, TARGET_SIGN[j1] * spread_command)  # Motor 18 (수정!)
                desired[j2] = self._clamp_target(j2, TARGET_SIGN[j2] * flex_command)  # Motor 19
                desired[j3] = self._clamp_target(j3, TARGET_SIGN[j3] * flex_command)  # Motor 20
            else:

                # 검지, 중지, 약지 (기존 로직)

                spread_deg = float(np.clip(

                    MOTION_RANGES['splay_default_gain'] * splay[finger],

                    -MOTION_RANGES['splay_default_limit'],

                    MOTION_RANGES['splay_default_limit']

                ))

                flex_deg = self._curl_to_flex_deg(curls[finger], MOTION_RANGES['flex_default'])

                spread_command = int(spread_deg * 10)

                flex_command = int(flex_deg * 10)

                desired[j0] = self._clamp_target(j0, TARGET_SIGN[j0] * spread_command)

                desired[j1] = self._clamp_target(j1, TARGET_SIGN[j1] * flex_command)

                desired[j2] = self._clamp_target(j2, TARGET_SIGN[j2] * flex_command)

                desired[j3] = self._clamp_target(j3, TARGET_SIGN[j3] * flex_command)

        return desired

    @staticmethod
    def _curl_to_flex_deg(curl_value: float, flex_degrees: float) -> float:
        """컬 값을 플렉스 각도로 변환"""
        curl_clamped = float(np.clip(curl_value, 0.0, 1.0))
        return curl_clamped * flex_degrees

    @staticmethod
    def _clamp_target(motor_id: int, target_0p1deg: int) -> int:
        """모터 제한값에 맞게 목표값 클램핑"""
        if motor_id not in MOTOR_LIMITS_DEG:
            return int(target_0p1deg)

        low, high = MOTOR_LIMITS_DEG[motor_id]
        return int(np.clip(int(target_0p1deg), int(low * 10), int(high * 10)))

    def apply_step_limits(self, desired: Dict[int, int], current_positions: Dict[int, int]) -> Dict[int, int]:
        """현재 위치 대비 스텝 제한 적용"""
        for motor_id in range(1, 21):
            max_step = int(MAX_STEP_DEG.get(motor_id, 25.0) * 10)
            current = int(current_positions.get(motor_id, 0))
            target = int(desired[motor_id])

            if target > current + max_step:
                desired[motor_id] = current + max_step
            elif target < current - max_step:
                desired[motor_id] = current - max_step

        return desired

    def apply_speed_limits(self, desired: Dict[int, int]) -> Dict[int, int]:
        """이전 목표값 대비 속도 제한 적용"""
        target = {}
        for motor_id in range(1, 21):
            max_speed = MAX_SPEED_DEG_S.get(motor_id, 120.0)
            max_delta = int(max_speed * 10.0 * DT)
            if max_delta < 1:
                max_delta = 1

            delta = int(desired[motor_id]) - int(self.previous_target[motor_id])

            if delta > max_delta:
                target[motor_id] = int(self.previous_target[motor_id]) + max_delta
            elif delta < -max_delta:
                target[motor_id] = int(self.previous_target[motor_id]) - max_delta
            else:
                target[motor_id] = int(desired[motor_id])

            self.previous_target[motor_id] = target[motor_id]

        return target

    def apply_base_angle_correction(self, positions: Dict[int, int]) -> Dict[int, int]:
        """
        센서 읽기값에 Base Angle 보정 적용

        Args:
            positions: 센서에서 읽은 원시 위치값 (0.1도 단위)

        Returns:
            보정된 위치값 (0.1도 단위)
        """
        corrected = {}
        for motor_id in range(1, 21):
            if motor_id in positions:
                raw_value = positions[motor_id]  # 0.1도 단위
                base_offset = int(BASE_ANGLES.get(motor_id, 0.0) * 10)  # 도 → 0.1도
                corrected[motor_id] = raw_value + base_offset

        return corrected

    def compute_raw_duty(self, target: Dict[int, int], current_positions: Dict[int, int]) -> Dict[int, int]:
        """위치 오차로부터 원시 듀티 사이클 계산"""
        raw_duty = {}

        for finger in FINGER_ORDER:
            j0, j1, j2, j3 = JOINT_MAP[finger]

            # 벌림 모터 처리 (모든 손가락 공통 - j0)
            error_0 = target[j0] - current_positions.get(j0, 0)

            if finger == "finger1":  # 엄지 벌림 (Motor 1)
                raw_duty[j0] = self._calculate_duty(error_0, GAINS['thumb_spread'],
                                                    DUTY_LIMITS['thumb_spread'], j0)
            elif finger == "finger5":  # 새끼 벌림 (Motor 17)
                raw_duty[j0] = self._calculate_duty(error_0, GAINS['pinky_spread'],
                                                    DUTY_LIMITS['pinky_spread'], j0)
            else:  # 검지, 중지, 약지 벌림 (Motor 5, 9, 13)
                raw_duty[j0] = self._calculate_duty(error_0, GAINS['spread'],
                                                    DUTY_LIMITS['spread'], j0)

            # 굽힘/특수 모터들 처리 (j1, j2, j3)
            if finger == "finger1":  # 엄지 특별 처리
                # Motor 2: 대립 (Opposition)
                error_1 = target[j1] - current_positions.get(j1, 0)
                raw_duty[j1] = self._calculate_duty(error_1, GAINS['thumb_opposition'],
                                                    DUTY_LIMITS['thumb_opposition'], j1)

                # Motor 3: MCP 굽힘
                error_2 = target[j2] - current_positions.get(j2, 0)
                raw_duty[j2] = self._calculate_duty(error_2, GAINS['thumb_mcp'],
                                                    DUTY_LIMITS['thumb_mcp'], j2)

                # Motor 4: IP 굽힘
                error_3 = target[j3] - current_positions.get(j3, 0)
                raw_duty[j3] = self._calculate_duty(error_3, GAINS['flex'],
                                                    DUTY_LIMITS['flex'], j3)

            elif finger == "finger5":  # 새끼 특별 처리
                # Motor 18 (j1): 내/외전 - 벌림 계열로 처리
                error_1 = target[j1] - current_positions.get(j1, 0)
                raw_duty[j1] = self._calculate_duty(error_1, GAINS['pinky_spread'],
                                                    DUTY_LIMITS['pinky_spread'], j1)

                # Motor 19, 20 (j2, j3): 굽힘
                error_2 = target[j2] - current_positions.get(j2, 0)
                error_3 = target[j3] - current_positions.get(j3, 0)
                raw_duty[j2] = self._calculate_duty(error_2, GAINS['flex'],
                                                    DUTY_LIMITS['flex'], j2)
                raw_duty[j3] = self._calculate_duty(error_3, GAINS['flex'],
                                                    DUTY_LIMITS['flex'], j3)

            else:  # 검지, 중지, 약지 (일반 손가락들) - 핵심 수정!
                # Motor j1, j2, j3: 모두 굽힘
                for joint in [j1, j2, j3]:
                    error = target[joint] - current_positions.get(joint, 0)
                    raw_duty[joint] = self._calculate_duty(error, GAINS['flex'],
                                                           DUTY_LIMITS['flex'], joint)

        return raw_duty

    @staticmethod
    def _calculate_duty(error_0p1deg: int, gain: float, limit: int, motor_id: int) -> int:
        """위치 오차를 듀티 사이클로 변환"""
        if abs(error_0p1deg) < DEADBAND_0P1DEG:
            return 0

        duty = int(gain * error_0p1deg)
        duty_clamped = int(np.clip(duty, -limit, limit))

        return DUTY_SIGN[motor_id] * duty_clamped

    def apply_duty_slew(self, raw_duty: Dict[int, int]) -> Dict[int, int]:
        """듀티 사이클에 슬루 레이트 제한 적용"""
        final_duty = {}
        for motor_id in range(1, 21):
            previous = int(self.previous_duty.get(motor_id, 0))
            new_duty = int(raw_duty.get(motor_id, 0))

            if new_duty > previous + MAX_DUTY_STEP:
                new_duty = previous + MAX_DUTY_STEP
            elif new_duty < previous - MAX_DUTY_STEP:
                new_duty = previous - MAX_DUTY_STEP

            self.previous_duty[motor_id] = new_duty
            final_duty[motor_id] = new_duty

        return final_duty

    def enforce_motor_enable_mask(self, current_positions: Dict[int, int], **data_dicts):
        """모터 활성화/비활성화 마스크 적용"""
        for motor_id in range(1, 21):
            if self.motor_enabled[motor_id]:
                continue

            hold_position = int(current_positions.get(motor_id, 0))

            for key, data_dict in data_dicts.items():
                if data_dict is not None and motor_id in data_dict:
                    if key in ['desired', 'target']:
                        data_dict[motor_id] = hold_position
                    elif key in ['raw', 'duty']:
                        data_dict[motor_id] = 0

    def enforce_isolate_mode(self, current_positions: Dict[int, int], **data_dicts):
        """격리 모드 적용 (단일 모터만 활성화)"""
        if not self.isolate_mode or self.isolate_motor_id is None:
            return

        for motor_id in range(1, 21):
            if motor_id == self.isolate_motor_id:
                continue

            hold_position = int(current_positions.get(motor_id, 0))

            for key, data_dict in data_dicts.items():
                if data_dict is not None and motor_id in data_dict:
                    if key in ['desired', 'target']:
                        data_dict[motor_id] = hold_position
                    elif key in ['raw', 'duty']:
                        data_dict[motor_id] = 0
