Tesollo/
├── __init__.py
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지 목록
├── config/
│   ├── __init__.py
│   └── constants.py       # 모든 설정값과 상수
├── hardware/
│   ├── __init__.py
│   ├── gripper_client.py  # TCP 통신 클라이언트
│   └── ft_sensor_client.py # F/T sensor 측정값
├── vision/
│   ├── __init__.py
│   └── hand_tracker.py    # MediaPipe 핸드 트래킹
├── control/
│   ├── __init__.py
│   ├── motor_controller.py # 모터 제어 로직
│   └── safety.py          # 안전 장치 및 제한
└── ui/
    ├── __init__.py
    ├── visualization.py    # HUD 및 시각화
    └── ft_dashboard.py    #ft_sensor 시각화
