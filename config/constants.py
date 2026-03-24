#constants.py
"""
DG-5F-M 그리퍼 제어를 위한 설정 상수들
"""
import numpy as np

# 장치 설정
GRIPPER_IP = "169.254.186.73"
GRIPPER_PORT = 502
MODEL_PATH = "hand_landmarker.task"

# 제어 루프 파라미터
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

# 제어 게인
GAINS = {
    'flex': 0.8,
    'spread': 1.2,
    'thumb_spread': 1.3,        # Motor 1용
    'thumb_opposition': 1.8,    # Motor 2용 ← 추가!
    'thumb_mcp': 1.0,
    'pinky_spread': 1.5,
    'pinky_flex': 1.1,
}


# 듀티 제한값
DUTY_LIMITS = {
    'flex': 200,
    'spread': 200,
    'thumb_spread': 200,
    'thumb_opposition': 200,
    'thumb_mcp': 200,
    'pinky_spread': 200,
    'pinky_flex': 200,
}


# 데드밴드 및 스무딩
DEADBAND_0P1DEG = 5
SMOOTH_ALPHA = 0.35
SPLAY_SMOOTH_ALPHA = 0.35

# 모션 범위 매핑
MOTION_RANGES = {
    'flex_default': 90.0,
    'flex_thumb_mcp': 220.0,
    'flex_thumb_ip': 130.0,
    'splay_default_gain': 1.0,
    'splay_default_limit': 25.0,
    'splay_thumb_gain': 2.0,
    'splay_thumb_limit': 70.0,       # Motor 1용 (안전하게 축소)
    'opposition_thumb_limit': 150.0, # Motor 2용 ← 추가!
}

# 손가락 및 모터 매핑
FINGER_ORDER = ["finger1", "finger2", "finger3", "finger4", "finger5"]

JOINT_MAP = {
    "finger1": [1, 2, 3, 4],      # 엄지: 1,2=spread, 3=MCP flex, 4=IP flex
    "finger2": [5, 6, 7, 8],      # 검지: 5=spread, 6/7/8=flex
    "finger3": [9, 10, 11, 12],   # 중지: 9=spread, 10/11/12=flex
    "finger4": [13, 14, 15, 16],  # 약지: 13=spread, 14/15/16=flex
    "finger5": [17, 18, 19, 20],  # 새끼: 17=spread, 18/19/20=flex
}


# 신호 매핑
# 기본은 +1, 반전이 필요한 모터만 별도 집합에서 관리
REVERSED_TARGET_MOTORS = {2, 3, 4, 17}

TARGET_SIGN = {
    motor_id: (-1 if motor_id in REVERSED_TARGET_MOTORS else 1)
    for motor_id in range(1, 21)
}

DUTY_SIGN = {i: 1 for i in range(1, 21)}


# 모터 각도 제한 (매뉴얼 기준)
MOTOR_LIMITS_DEG = {
    1: (-20, 0),      # Motor 1: 엄지 벌림
    2: (20, 100),      # Motor 2: 엄지 대립
    3: (-150, 0),     # Motor 3: 엄지 MCP
    4: (-90, 90),      # Motor 4: 엄지 IP
    5: (-2, 20),      # Motor 5: 검지 벌림
    6: (0, 115),       # Motor 6: 검지 굽힘1
    7: (-90, 90),      # Motor 7: 검지 굽힘2
    8: (-90, 90),      # Motor 8: 검지 굽힘3
    9: (-30, 30),      # Motor 9: 중지 벌림
    10: (0, 115),      # Motor 10: 중지 굽힘1
    11: (-90, 90),     # Motor 11: 중지 굽힘2
    12: (-90, 90),     # Motor 12: 중지 굽힘3
    13: (-32, 15),     # Motor 13: 약지 벌림
    14: (0, 110),      # Motor 14: 약지 굽힘1
    15: (-90, 90),     # Motor 15: 약지 굽힘2
    16: (-90, 90),     # Motor 16: 약지 굽힘3
    17: (0, 0),      # Motor 17: 새끼 벌림 (매뉴얼 기준)
    18: (-15, -5),     # Motor 18: 새끼 굽힘1 (매뉴얼 기준)
    19: (-90, 90),     # Motor 19: 새끼 굽힘2
    20: (-90, 90),     # Motor 20: 새끼 굽힘3
}

# 모터별 Base Angle (영점 보정값) - 단위: 도
# 하드웨어 조립 시 발생하는 기계적 오프셋 보정
# 양수: 센서값이 실제보다 작을 때 (시계방향 보정)
# 음수: 센서값이 실제보다 클 때 (반시계방향 보정)
BASE_ANGLES = {
    1: 0.0,    # Motor 1: 엄지 벌림
    2: 0.0,    # Motor 2: 엄지 대립
    3: 20.0,    # Motor 3: 엄지 MCP
    4: 0.0,    # Motor 4: 엄지 IP
    5: 0.0,    # Motor 5: 검지 벌림
    6: 0.0,    # Motor 6: 검지 굽힘1
    7: 0.0,    # Motor 7: 검지 굽힘2
    8: 0.0,    # Motor 8: 검지 굽힘3
    9: -7.0,    # Motor 9: 중지 벌림
    10: 0.0,   # Motor 10: 중지 굽힘1
    11: 0.0,   # Motor 11: 중지 굽힘2
    12: 0.0,   # Motor 12: 중지 굽힘3
    13: 0.0,   # Motor 13: 약지 벌림
    14: 0.0,   # Motor 14: 약지 굽힘1
    15: 0.0,   # Motor 15: 약지 굽힘2
    16: 0.0,   # Motor 16: 약지 굽힘3
    17: 0.0,   # Motor 17: 새끼 벌림
    18: 0.0,   # Motor 18: 새끼 굽힘1
    19: 0.0,   # Motor 19: 새끼 굽힘2
    20: 0.0,   # Motor 20: 새끼 굽힘3
}


# 속도 제한 (도/초)
MAX_SPEED_DEG_S = {m: 80.0 for m in range(1, 21)}
for m in [1, 5, 9, 13, 17]:
    MAX_SPEED_DEG_S[m] = 50.0
MAX_SPEED_DEG_S.update({2: 50.0, 3: 70.0, 4: 90.0})

# 스텝 제한
MAX_STEP_DEG = {m: 25.0 for m in range(1, 21)}
for m in [1, 5, 9, 13, 17]:
    MAX_STEP_DEG[m] = 10.0
MAX_STEP_DEG.update({2: 10.0, 3: 20.0, 4: 20.0})

# 듀티 슬루 제한
MAX_DUTY_STEP = 10

# 전역 부하 제한
GLOBAL_LIMITS = {
    'total_duty_budget': 1500,
    'max_active_joints': 10,
    'min_duty_to_move': 20,
    'protected_joints': {17, 18},
}

# 모터 역할 이름
MOTOR_ROLES = {
    1: "thumb spread A", 2: "thumb spread B", 3: "thumb MCP flex", 4: "thumb IP flex",
    5: "index spread", 6: "index flex 1", 7: "index flex 2", 8: "index flex 3",
    9: "middle spread", 10: "middle flex 1", 11: "middle flex 2", 12: "middle flex 3",
    13: "ring spread", 14: "ring flex 1", 15: "ring flex 2", 16: "ring flex 3",
    17: "pinky spread base", 18: "pinky adduction/abduction", 19: "pinky flex 1", 20: "pinky flex 2"
}

# 핸드 트래킹 랜드마크
FINGER_LANDMARKS = {
    "finger1": [1, 2, 3, 4],
    "finger2": [5, 6, 7, 8],
    "finger3": [9, 10, 11, 12],
    "finger4": [13, 14, 15, 16],
    "finger5": [17, 18, 19, 20],
}
