# ft_dashboard.py
"""
Professional F/T Sensor Real-time Dashboard
Tesollo Hand Teleoperation System용 전문 F/T 센서 모니터링
"""

import cv2
import numpy as np
from collections import deque
import time

# Constants import with fallback
try:
    from config.constants import (
        MAX_FORCE_LIMIT, MAX_TORQUE_LIMIT,
        FORCE_WARNING_THRESHOLD, TORQUE_WARNING_THRESHOLD
    )
except ImportError:
    print("⚠ Warning: Using default F/T limits in dashboard")
    MAX_FORCE_LIMIT = 50.0
    MAX_TORQUE_LIMIT = 5.0
    FORCE_WARNING_THRESHOLD = 40.0
    TORQUE_WARNING_THRESHOLD = 4.0

class FTDashboard:
    def __init__(self, width=1000, height=700, history_length = 300):
        """
        대시보드 초기화

        Args:
            width, height: 창 크기
            history_length: 그래프에 표시할 데이터 포인트 수 (300 = 약 10초)
        """
        self.window_name = "Tesollo F/T Sensor Dashboard"
        self.width = width
        self.height = height
        self.history_length = history_length

        # 데이터 히스토리 저장 (실시간 그래프용)
        self.force_history = [deque(maxlen=history_length) for _ in range(3)]  # Fx, Fy, Fz
        self.torque_history = [deque(maxlen=history_length) for _ in range(3)]  # Tx, Ty, Tz
        self.magnitude_history = {
            'force': deque(maxlen=history_length),
            'torque': deque(maxlen=history_length)
        }

        # 현재 센서 값
        self.current_force = np.zeros(3)
        self.current_torque = np.zeros(3)
        self.force_magnitude = 0.0
        self.torque_magnitude = 0.0

        # 통계 데이터
        self.max_force_recorded = 0.0
        self.max_torque_recorded = 0.0
        self.start_time = time.time()

        # 경고 상태
        self.force_warning = False
        self.torque_warning = False
        self.emergency_state = False

        # 시각화 설정
        self.colors = {
            'force': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # R, G, B for Fx, Fy, Fz
            'torque': [(0, 128, 255), (255, 0, 255), (255, 255, 0)],  # Orange, Magenta, Cyan
            'magnitude': (0, 255, 255),  # Cyan
            'warning': (0, 255, 255),  # Yellow
            'danger': (0, 0, 255),  # Red
            'safe': (0, 255, 0)  # Green
        }

        self.labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        self.units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

        # 스레딩 제어
        self.running = False
        self.display_thread: Optional[threading.Thread] = None
        self.data_lock = threading.Lock()

    def start(self):
        """별도 스레드에서 대시보드 창 시작"""
        if self.running:
            print("⚠ F/T Dashboard already running")
            return

        self.running = True
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="FTDashboard"
        )
        self.display_thread.start()
        print("✓ F/T Dashboard started in separate window")

    def stop(self):
        """대시보드 종료"""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        cv2.destroyWindow(self.window_name)
        print("✓ F/T Dashboard stopped")

    def update_data(self, force: np.ndarray, torque: np.ndarray):
        """
        F/T 센서 데이터 업데이트 (메인 루프에서 호출)

        Args:
            force: [Fx, Fy, Fz] 힘 벡터 (N)
            torque: [Tx, Ty, Tz] 토크 벡터 (Nm)
        """
        with self.data_lock:
            # 현재 값 저장
            self.current_force = force.copy()
            self.current_torque = torque.copy()

            # 크기 계산
            self.force_magnitude = np.linalg.norm(force)
            self.torque_magnitude = np.linalg.norm(torque)

            # 히스토리 업데이트
            for i in range(3):
                self.force_history[i].append(force[i])
                self.torque_history[i].append(torque[i])

            self.magnitude_history['force'].append(self.force_magnitude)
            self.magnitude_history['torque'].append(self.torque_magnitude)

            # 통계 업데이트
            self.max_force_recorded = max(self.max_force_recorded, self.force_magnitude)
            self.max_torque_recorded = max(self.max_torque_recorded, self.torque_magnitude)

            # 경고 상태 확인
            self.force_warning = self.force_magnitude > FORCE_WARNING_THRESHOLD
            self.torque_warning = self.torque_magnitude > TORQUE_WARNING_THRESHOLD
            self.emergency_state = (self.force_magnitude > MAX_FORCE_LIMIT or
                                    self.torque_magnitude > MAX_TORQUE_LIMIT)

    def _display_loop(self):
        """대시보드 표시 루프 (별도 스레드)"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        while self.running:
            # 캔버스 생성 (어두운 배경)
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas.fill(20)  # 어두운 회색 배경

            with self.data_lock:
                # 대시보드 구성 요소들 그리기
                self._draw_header(canvas)
                self._draw_current_values(canvas)
                self._draw_bar_gauges(canvas)
                self._draw_time_series_graphs(canvas)
                self._draw_3d_vector_display(canvas)
                self._draw_statistics_panel(canvas)
                self._draw_status_indicators(canvas)

            cv2.imshow(self.window_name, canvas)

            # 키 입력 처리 (옵션)
            key = cv2.waitKey(33) & 0xFF  # 30fps
            if key == ord('q'):
                break
            elif key == ord('r'):  # 통계 리셋
                self._reset_statistics()

        cv2.destroyWindow(self.window_name)

    def _draw_header(self, canvas):
        """대시보드 헤더 그리기"""
        # 제목
        cv2.putText(canvas, "TESOLLO F/T SENSOR DASHBOARD",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 실행 시간
        runtime = time.time() - self.start_time
        cv2.putText(canvas, f"Runtime: {runtime:.1f}s",
                    (self.width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 구분선
        cv2.line(canvas, (20, 60), (self.width - 20, 60), (100, 100, 100), 2)

    def _draw_current_values(self, canvas):
        """현재 센서 값 표시"""
        start_y = 90
        col1_x, col2_x = 50, 300

        # 힘 값
        cv2.putText(canvas, "FORCE (N)", (col1_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i in range(3):
            y_pos = start_y + 30 + (i * 25)
            color = self.colors['force'][i]
            cv2.putText(canvas, f"{self.labels[i]}: {self.current_force[i]:8.2f}",
                        (col1_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 힘 크기
        mag_color = self._get_magnitude_color(self.force_magnitude, MAX_FORCE_LIMIT)
        cv2.putText(canvas, f"|F|: {self.force_magnitude:8.2f}",
                    (col1_x, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mag_color, 2)

        # 토크 값
        cv2.putText(canvas, "TORQUE (Nm)", (col2_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        for i in range(3):
            y_pos = start_y + 30 + (i * 25)
            color = self.colors['torque'][i]
            cv2.putText(canvas, f"{self.labels[i + 3]}: {self.current_torque[i]:8.3f}",
                        (col2_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 토크 크기
        mag_color = self._get_magnitude_color(self.torque_magnitude, MAX_TORQUE_LIMIT)
        cv2.putText(canvas, f"|T|: {self.torque_magnitude:8.3f}",
                    (col2_x, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mag_color, 2)

    def _draw_bar_gauges(self, canvas):
        """수평 바 게이지 그리기"""
        gauge_y = 240
        gauge_w = 300
        gauge_h = 20

        # 힘 게이지
        self._draw_single_gauge(canvas, (500, gauge_y), gauge_w, gauge_h,
                                self.force_magnitude, MAX_FORCE_LIMIT,
                                "FORCE MAGNITUDE", self.colors['magnitude'])

        # 토크 게이지
        self._draw_single_gauge(canvas, (500, gauge_y + 60), gauge_w, gauge_h,
                                self.torque_magnitude, MAX_TORQUE_LIMIT,
                                "TORQUE MAGNITUDE", self.colors['magnitude'])

    def _draw_single_gauge(self, canvas, pos, width, height, value, max_val, label, color):
        """개별 게이지 그리기"""
        x, y = pos

        # 라벨
        cv2.putText(canvas, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 배경 (게이지 틀)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (120, 120, 120), 2)

        # 값 비율 계산
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0.0
        fill_width = int(width * ratio)

        # 게이지 색상 결정
        if ratio > 0.9:
            gauge_color = self.colors['danger']
        elif ratio > 0.7:
            gauge_color = self.colors['warning']
        else:
            gauge_color = self.colors['safe']

        # 게이지 채우기
        if fill_width > 0:
            cv2.rectangle(canvas, (x + 2, y + 2),
                          (x + fill_width - 2, y + height - 2), gauge_color, -1)

        # 경계선들 (25%, 50%, 75%, 100%)
        for pct in [0.25, 0.5, 0.75, 1.0]:
            line_x = int(x + width * pct)
            line_color = (0, 0, 255) if pct == 1.0 else (100, 100, 100)
            cv2.line(canvas, (line_x, y), (line_x, y + height), line_color, 1)

        # 수치 표시
        cv2.putText(canvas, f"{value:.2f}/{max_val:.1f}",
                    (x + width + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_time_series_graphs(self, canvas):
        """시계열 그래프 그리기"""
        graph_start_y = 320
        graph_height = 120
        graph_width = 400

        # 힘 그래프
        self._draw_time_graph(canvas, (50, graph_start_y), graph_width, graph_height,
                              self.force_history, self.colors['force'], "FORCE HISTORY", "N")

        # 토크 그래프
        self._draw_time_graph(canvas, (500, graph_start_y), graph_width, graph_height,
                              self.torque_history, self.colors['torque'], "TORQUE HISTORY", "Nm")

        # 크기 그래프
        magnitude_data = [self.magnitude_history['force'], self.magnitude_history['torque']]
        magnitude_colors = [(0, 255, 255), (255, 0, 255)]  # Cyan, Magenta
        self._draw_time_graph(canvas, (275, graph_start_y + 140), graph_width, graph_height,
                              magnitude_data, magnitude_colors, "MAGNITUDE HISTORY", "")

    def _draw_time_graph(self, canvas, pos, width, height, data_series, colors, title, unit):
        """개별 시계열 그래프 그리기"""
        x, y = pos

        # 제목
        cv2.putText(canvas, title, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 배경
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)

        # 중앙선 (0점)
        mid_y = y + height // 2
        cv2.line(canvas, (x, mid_y), (x + width, mid_y), (60, 60, 60), 1)

        # 데이터 라인 그리기
        for series_idx, data in enumerate(data_series):
            if len(data) < 2:
                continue

            color = colors[series_idx] if isinstance(colors[0], tuple) else colors

            # 데이터 정규화 및 좌표 변환
            data_array = np.array(data)
            if len(data_array) == 0:
                continue

            # 자동 스케일링
            data_max = max(abs(data_array.max()), abs(data_array.min()), 0.1)

            points = []
            for i, val in enumerate(data_array):
                px = int(x + (i / self.history_length) * width)
                # Y 좌표 반전 (상단이 양수)
                py = int(mid_y - (val / data_max) * (height // 2 * 0.8))
                points.append([px, py])

            # 폴리라인 그리기
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points)], False, color, 2, cv2.LINE_AA)

    def _draw_3d_vector_display(self, canvas):
        """3D 벡터 표시 (2D 투영)"""
        center_x, center_y = self.width - 200, 450
        scale = 30

        # 배경 원
        cv2.circle(canvas, (center_x, center_y), 80, (40, 40, 40), -1)
        cv2.circle(canvas, (center_x, center_y), 80, (100, 100, 100), 2)

        # 축 그리기
        cv2.line(canvas, (center_x - 70, center_y), (center_x + 70, center_y), (60, 60, 60), 1)
        cv2.line(canvas, (center_x, center_y - 70), (center_x, center_y + 70), (60, 60, 60), 1)

        # 힘 벡터 (XY 평면 투영)
        if self.force_magnitude > 0.1:  # 노이즈 필터링
            fx_scaled = int(self.current_force[0] * scale / 10)
            fy_scaled = int(self.current_force[1] * scale / 10)

            end_x = center_x + fx_scaled
            end_y = center_y - fy_scaled  # Y축 반전

            # 벡터 화살표
            cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y),
                            (0, 255, 255), 3, tipLength=0.3)

        # 라벨
        cv2.putText(canvas, "FORCE VECTOR", (center_x - 60, center_y - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"|F|={self.force_magnitude:.1f}N", (center_x - 60, center_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_statistics_panel(self, canvas):
        """통계 정보 패널"""
        panel_x, panel_y = 50, 500
        panel_w, panel_h = 200, 150

        # 패널 배경
        cv2.rectangle(canvas, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.rectangle(canvas, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)

        # 제목
        cv2.putText(canvas, "STATISTICS", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 통계 정보
        stats = [
            f"Max Force: {self.max_force_recorded:.2f}N",
            f"Max Torque: {self.max_torque_recorded:.3f}Nm",
            f"Data Points: {len(self.magnitude_history['force'])}",
            f"Runtime: {time.time() - self.start_time:.1f}s"
        ]

        for i, stat in enumerate(stats):
            cv2.putText(canvas, stat, (panel_x + 10, panel_y + 55 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_status_indicators(self, canvas):
        """상태 표시등"""
        indicator_y = self.height - 50

        # 상태별 색상과 메시지
        if self.emergency_state:
            status_color = self.colors['danger']
            status_text = "🚨 EMERGENCY"
        elif self.force_warning or self.torque_warning:
            status_color = self.colors['warning']
            status_text = "⚠ WARNING"
        else:
            status_color = self.colors['safe']
            status_text = "✓ NORMAL"

        # 상태 표시
        cv2.circle(canvas, (50, indicator_y), 15, status_color, -1)
        cv2.putText(canvas, status_text, (80, indicator_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 조작 안내
        cv2.putText(canvas, "Press 'Q' to quit, 'R' to reset stats",
                    (self.width - 350, indicator_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _get_magnitude_color(self, magnitude, limit):
        """크기에 따른 색상 결정"""
        ratio = magnitude / limit if limit > 0 else 0
        if ratio > 0.9:
            return self.colors['danger']
        elif ratio > 0.7:
            return self.colors['warning']
        else:
            return self.colors['safe']

    def _reset_statistics(self):
        """통계 데이터 초기화"""
        with self.data_lock:
            self.max_force_recorded = 0.0
            self.max_torque_recorded = 0.0
            self.start_time = time.time()
        print("✓ F/T Dashboard statistics reset")

    def get_dashboard_status(self):
        """대시보드 상태 정보 반환"""
        return {
            'running': self.running,
            'emergency_state': self.emergency_state,
            'force_warning': self.force_warning,
            'torque_warning': self.torque_warning,
            'max_force_recorded': self.max_force_recorded,
            'max_torque_recorded': self.max_torque_recorded
        }

    def get_dashboard_frame(self):
        """메인 스레드에서 사용할 이미지 프레임 반환"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas.fill(25)

        # 대시보드 요소들 그리기
        self._draw_header(canvas)
        self._draw_current_values(canvas)
        self._draw_gauges(canvas)
        self._draw_status(canvas)

        return canvas


# 독립 실행 테스트용
if __name__ == "__main__":
    """대시보드 단독 테스트"""
    dashboard = FTDashboard()
    dashboard.start()

    try:
        # 시뮬레이션 데이터로 테스트
        t = 0
        while True:
            # 가상 F/T 데이터 생성
            force = np.array([
                15 * np.sin(t * 0.5) + 5 * np.random.normal(),
                10 * np.cos(t * 0.3) + 3 * np.random.normal(),
                20 + 8 * np.sin(t * 0.2) + 2 * np.random.normal()
            ])
            torque = np.array([
                2 * np.sin(t * 0.7) + 0.5 * np.random.normal(),
                1.5 * np.cos(t * 0.4) + 0.3 * np.random.normal(),
                1 + 0.8 * np.sin(t * 0.6) + 0.2 * np.random.normal()
            ])

            dashboard.update_data(force, torque)
            time.sleep(0.033)  # 30Hz
            t += 0.033

    except KeyboardInterrupt:
        print("\nStopping dashboard test...")
        dashboard.stop()
