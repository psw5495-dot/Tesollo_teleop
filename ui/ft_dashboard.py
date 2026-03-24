# ft_dashboard.py - 수정된 버전

import cv2
import numpy as np
from collections import deque
import time

# 안전한 constants import (에러 방지)
try:
    from config.constants import (
        MAX_FORCE_LIMIT, MAX_TORQUE_LIMIT,
        FORCE_WARNING_THRESHOLD, TORQUE_WARNING_THRESHOLD
    )
except ImportError:
    print("⚠ Warning: Using default F/T limits")
    MAX_FORCE_LIMIT = 50.0
    MAX_TORQUE_LIMIT = 5.0
    FORCE_WARNING_THRESHOLD = 40.0
    TORQUE_WARNING_THRESHOLD = 4.0

class FTDashboard:
    def __init__(self, width=1000, height=700, history_length=300):
        self.width = width
        self.height = height
        self.history_length = history_length
        
        # 데이터 저장소
        self.force_history = [deque(maxlen=history_length) for _ in range(3)]
        self.torque_history = [deque(maxlen=history_length) for _ in range(3)]
        self.magnitude_history = {
            'force': deque(maxlen=history_length),
            'torque': deque(maxlen=history_length)
        }
        
        # 현재 상태
        self.current_force = np.zeros(3)
        self.current_torque = np.zeros(3)
        self.force_magnitude = 0.0
        self.torque_magnitude = 0.0
        
        # 통계
        self.max_force_recorded = 0.0
        self.max_torque_recorded = 0.0
        self.start_time = time.time()
        
        # 경고 상태
        self.emergency_state = False
        
        # 색상 정의 (BGR 포맷)
        self.colors = {
            'force': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],     # Red, Green, Blue
            'torque': [(0, 128, 255), (255, 0, 255), (255, 255, 0)], # Orange, Magenta, Cyan
            'warning': (0, 255, 255),  # Yellow
            'danger': (0, 0, 255),     # Red  
            'safe': (0, 255, 0)        # Green
        }

    def update_data(self, force, torque):
        """F/T 데이터 업데이트"""
        self.current_force = force.copy()
        self.current_torque = torque.copy()
        
        self.force_magnitude = np.linalg.norm(force)
        self.torque_magnitude = np.linalg.norm(torque)
        
        # 히스토리 저장
        for i in range(3):
            self.force_history[i].append(force[i])
            self.torque_history[i].append(torque[i])
            
        self.magnitude_history['force'].append(self.force_magnitude)
        self.magnitude_history['torque'].append(self.torque_magnitude)
        
        # 통계 업데이트
        self.max_force_recorded = max(self.max_force_recorded, self.force_magnitude)
        self.max_torque_recorded = max(self.max_torque_recorded, self.torque_magnitude)
        
        # 비상 상태 확인
        self.emergency_state = (self.force_magnitude > MAX_FORCE_LIMIT or 
                               self.torque_magnitude > MAX_TORQUE_LIMIT)

    def get_dashboard_frame(self):
        """
        [핵심 수정] 스레드에서 imshow 하지 않고, 이미지만 반환
        메인 루프에서 이 이미지를 받아서 cv2.imshow() 호출
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas.fill(25)  # 어두운 배경
        
        # 대시보드 구성 요소 그리기
        self._draw_header(canvas)
        self._draw_current_values(canvas)
        self._draw_gauges(canvas)
        self._draw_graphs(canvas)
        self._draw_status(canvas)
        
        return canvas

    def _draw_header(self, canvas):
        """헤더 그리기"""
        cv2.putText(canvas, "TESOLLO F/T SENSOR DASHBOARD", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        runtime = time.time() - self.start_time
        cv2.putText(canvas, f"Runtime: {runtime:.1f}s", 
                   (self.width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _draw_current_values(self, canvas):
        """현재 센서 값 표시"""
        # Force 값
        cv2.putText(canvas, "FORCE (N)", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        labels = ['Fx', 'Fy', 'Fz']
        for i in range(3):
            text = f"{labels[i]}: {self.current_force[i]:7.2f}"
            cv2.putText(canvas, text, (50, 120 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['force'][i], 2)
        
        # Torque 값
        cv2.putText(canvas, "TORQUE (Nm)", (300, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        labels_t = ['Tx', 'Ty', 'Tz']
        for i in range(3):
            text = f"{labels_t[i]}: {self.current_torque[i]:7.3f}"
            cv2.putText(canvas, text, (300, 120 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['torque'][i], 2)

    def _draw_gauges(self, canvas):
        """수평 게이지 바 그리기"""
        # Force Magnitude Gauge
        force_ratio = min(self.force_magnitude / MAX_FORCE_LIMIT, 1.0)
        force_color = self.colors['danger'] if force_ratio > 0.8 else self.colors['safe']
        
        cv2.putText(canvas, f"Force Magnitude: {self.force_magnitude:.1f}N / {MAX_FORCE_LIMIT}N", 
                   (550, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 게이지 배경
        cv2.rectangle(canvas, (550, 120), (900, 145), (50, 50, 50), -1)
        # 게이지 채우기
        fill_width = int(350 * force_ratio)
        cv2.rectangle(canvas, (550, 120), (550 + fill_width, 145), force_color, -1)
        # 게이지 테두리
        cv2.rectangle(canvas, (550, 120), (900, 145), (150, 150, 150), 2)
        
        # Torque Magnitude Gauge  
        torque_ratio = min(self.torque_magnitude / MAX_TORQUE_LIMIT, 1.0)
        torque_color = self.colors['danger'] if torque_ratio > 0.8 else self.colors['safe']
        
        cv2.putText(canvas, f"Torque Magnitude: {self.torque_magnitude:.2f}Nm / {MAX_TORQUE_LIMIT}Nm", 
                   (550, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(canvas, (550, 180), (900, 205), (50, 50, 50), -1)
        fill_width = int(350 * torque_ratio)
        cv2.rectangle(canvas, (550, 180), (550 + fill_width, 205), torque_color, -1)
        cv2.rectangle(canvas, (550, 180), (900, 205), (150, 150, 150), 2)

    def _draw_graphs(self, canvas):
        """시계열 그래프 그리기"""
        graph_x, graph_y = 50, 250
        graph_w, graph_h = 400, 120
        
        # 그래프 배경
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 1)
        cv2.putText(canvas, "Force Magnitude History", (graph_x + 5, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 그래프 그리기
        if len(self.magnitude_history['force']) > 1:
            points = []
            max_val = max(max(self.magnitude_history['force']), MAX_FORCE_LIMIT * 0.1)
            
            for i, val in enumerate(self.magnitude_history['force']):
                x = graph_x + int((i / len(self.magnitude_history['force'])) * graph_w)
                y = graph_y + graph_h - int((val / max_val) * graph_h * 0.8)
                points.append([x, y])
            
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points)], False, (0, 255, 255), 2)

    def _draw_status(self, canvas):
        """상태 표시"""
        status_y = self.height - 50
        
        if self.emergency_state:
            status_color = self.colors['danger']
            status_text = "🚨 EMERGENCY - LIMITS EXCEEDED"
        elif (self.force_magnitude > FORCE_WARNING_THRESHOLD or 
              self.torque_magnitude > TORQUE_WARNING_THRESHOLD):
            status_color = self.colors['warning']
            status_text = "⚠ WARNING - HIGH VALUES"
        else:
            status_color = self.colors['safe']
            status_text = "✓ NORMAL OPERATION"
        
        cv2.putText(canvas, status_text, (50, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 통계 정보
        cv2.putText(canvas, f"Max Force: {self.max_force_recorded:.1f}N  Max Torque: {self.max_torque_recorded:.2f}Nm", 
                   (self.width - 400, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def get_dashboard_status(self):
        """상태 정보 반환"""
        return {
            'running': True,
            'emergency_state': self.emergency_state,
            'max_force_recorded': self.max_force_recorded,
            'max_torque_recorded': self.max_torque_recorded
        }

    def start(self):
        """호환성을 위한 더미 메서드"""
        pass
        
    def stop(self):
        """호환성을 위한 더미 메서드"""
        pass
