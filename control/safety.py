#safety.py
"""
안전 장치 및 부하 관리 기능
"""
from typing import Dict
import logging

from config.constants import GLOBAL_LIMITS

logger = logging.getLogger(__name__)


def make_zero_duty() -> Dict[int, int]:
    """모든 모터의 듀티를 0으로 설정하는 딕셔너리 생성"""
    return {joint_id: 0 for joint_id in range(1, 21)}


def apply_global_limits(raw_duty_dict: Dict[int, int]) -> Dict[int, int]:
    """
    전역 부하 제한 적용:
    1. 작은 듀티 값 제거
    2. 동시 활성 관절 수 제한 (Top-K + 보호 관절)
    3. 총 듀티를 예산 내로 스케일링
    """
    duty = dict(raw_duty_dict)

    # 1. 최소 듀티 이하 값 제거
    removed_count = 0
    for motor_id in list(duty.keys()):
        if abs(duty[motor_id]) < GLOBAL_LIMITS['min_duty_to_move']:
            duty[motor_id] = 0
            removed_count += 1

    if removed_count > 0:
        logger.debug(f"최소 듀티 이하 {removed_count}개 모터 비활성화")

    # 2. 활성 관절 수 제한
    active_motors = [(motor_id, abs(value)) for motor_id, value in duty.items() if value != 0]

    if len(active_motors) > GLOBAL_LIMITS['max_active_joints']:
        # 듀티 크기순으로 정렬
        active_motors.sort(key=lambda x: x[1], reverse=True)

        # 보호 관절 우선 유지
        keep_motors = set(motor_id for motor_id in GLOBAL_LIMITS['protected_joints']
                          if duty.get(motor_id, 0) != 0)

        # Top-K 선택
        for motor_id, _ in active_motors:
            if len(keep_motors) >= GLOBAL_LIMITS['max_active_joints']:
                break
            keep_motors.add(motor_id)

        # 선택되지 않은 모터들 비활성화
        deactivated_count = 0
        for motor_id in list(duty.keys()):
            if motor_id not in keep_motors:
                if duty[motor_id] != 0:
                    deactivated_count += 1
                duty[motor_id] = 0

        if deactivated_count > 0:
            logger.debug(f"Top-K 제한으로 {deactivated_count}개 모터 비활성화")

    # 3. 총 듀티 예산 제한
    total_duty = sum(abs(value) for value in duty.values())

    if total_duty > GLOBAL_LIMITS['total_duty_budget'] and total_duty > 0:
        scale_factor = GLOBAL_LIMITS['total_duty_budget'] / total_duty

        for motor_id in list(duty.keys()):
            duty[motor_id] = int(duty[motor_id] * scale_factor)

        logger.debug(f"듀티 예산 초과로 {scale_factor:.3f} 배율 적용")

    return duty


def check_emergency_conditions(current_positions: Dict[int, int],
                               target_positions: Dict[int, int]) -> bool:
    """
    비상 상황 감지

    Returns:
        True if emergency condition detected
    """
    # 예시: 큰 위치 오차 감지
    max_error = 0
    for motor_id in range(1, 21):
        current = current_positions.get(motor_id, 0)
        target = target_positions.get(motor_id, 0)
        error = abs(current - target)
        max_error = max(max_error, error)

    # 50도 이상 오차 시 비상 상황
    if max_error > 500:  # 0.1도 단위
        logger.warning(f"큰 위치 오차 감지: {max_error / 10.0}도")
        return True

    return False
