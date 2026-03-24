"""
MediaPipe Hand Tracker - Tasks API Version
High-performance implementation with .task model file
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from typing import Optional, Dict
import os

# Constants import with fallback
try:
    from config.constants import (
        HAND_LANDMARKER_MODEL_PATH,
        MAX_NUM_HANDS,
        MIN_HAND_DETECTION_CONFIDENCE,
        MIN_HAND_PRESENCE_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE
    )
except ImportError:
    # 기본값 사용
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HAND_LANDMARKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
    MAX_NUM_HANDS = 1
    MIN_HAND_DETECTION_CONFIDENCE = 0.5
    MIN_HAND_PRESENCE_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5


class HandTracker:
    """
    MediaPipe Tasks API 기반 손 추적
    .task 모델 파일 사용 - 고성능, 최신 기능
    """
    
    def __init__(self, model_path: str = HAND_LANDMARKER_MODEL_PATH):
        """
        Tasks API 손 추적기 초기화
        
        Args:
            model_path: .task 모델 파일 경로
        """
        print(f"Initializing MediaPipe Hand Tracker (Tasks API)...")
        print(f"Model path: {model_path}")
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                f"Please download the model file and place it in the models/ folder.\n"
                f"Download URL: https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        # Tasks API 옵션 설정
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # 실시간 비디오 처리 최적화
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # HandLandmarker 생성
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # 시각화용 유틸리티
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 손가락 관절 인덱스 (기존과 동일)
        self.finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        self.finger_tips = [4, 8, 12, 16, 20]
        
        # 통계 및 타이밍
        self.frame_count = 0
        self.detection_count = 0
        self.timestamp_ms = 0
        
        print("✓ HandTracker (Tasks API) initialized successfully")
    
    def process(self, frame: np.ndarray) -> Optional[Dict]:
        """
        프레임에서 손 감지 및 데이터 추출
        
        Args:
            frame: BGR 이미지 (OpenCV 형식)
            
        Returns:
            손 데이터 딕셔너리 또는 None
        """
        if frame is None or frame.size == 0:
            return None
        
        self.frame_count += 1
        self.timestamp_ms += 33  # 약 30fps 가정
        
        # BGR → RGB 변환 (MediaPipe 필수)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image 객체 생성
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Tasks API로 손 감지 (타임스탬프 기반)
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        # 손이 감지되지 않음
        if not detection_result.hand_landmarks:
            if self.frame_count % 120 == 1:
                print(f"[HandTracker] No hand detected (frame {self.frame_count})")
            return None
        
        # 첫 번째 손 사용
        hand_landmarks = detection_result.hand_landmarks[0]
        hand_handedness = detection_result.handedness[0] if detection_result.handedness else None
        
        self.detection_count += 1
        
        # 화면에 랜드마크 그리기
        self._draw_landmarks(frame, hand_landmarks)
        
        # 손 데이터 추출
        hand_data = self._extract_hand_data(hand_landmarks, hand_handedness, frame.shape)
        
        # 주기적 통계 출력
        if self.detection_count % 60 == 1:
            detection_rate = (self.detection_count / self.frame_count) * 100
            print(f"[HandTracker] Detection rate: {detection_rate:.1f}% - Tasks API active!")
        
        return hand_data
    
    def _draw_landmarks(self, frame: np.ndarray, hand_landmarks):
        """
        프레임에 손 랜드마크 그리기
        Tasks API 결과를 Solutions API 호환 형식으로 변환
        """
        h, w = frame.shape[:2]
        
        # NormalizedLandmark 리스트를 protobuf 형식으로 변환
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in hand_landmarks
        ])
        
        # MediaPipe 표준 스타일로 그리기
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_proto,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # 손가락 끝에 강조 표시
        for tip_idx in self.finger_tips:
            lm = hand_landmarks[tip_idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)  # 초록색 원
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)  # 흰색 테두리
    
    def _extract_hand_data(self, hand_landmarks, hand_handedness, frame_shape) -> Dict:
        """랜드마크에서 제어 데이터 추출"""
        h, w = frame_shape[:2]
        
        # 랜드마크를 픽셀 좌표로 변환
        landmarks_px = []
        for lm in hand_landmarks:
            landmarks_px.append({
                'x': lm.x * w,
                'y': lm.y * h,
                'z': lm.z
            })
        
        # 손가락 굽힘 계산
        finger_states = self._calculate_finger_states(landmarks_px)
        
        # 손 방향 정보 (Tasks API는 Category 객체 반환)
        hand_label = hand_handedness[0].category_name if hand_handedness else "Unknown"
        hand_score = hand_handedness[0].score if hand_handedness else 0.0
        
        return {
            'landmarks': landmarks_px,
            'finger_states': finger_states,
            'hand_label': hand_label,
            'hand_score': hand_score,
            'detected': True
        }
    
    def _calculate_finger_states(self, landmarks_px) -> Dict[str, float]:
        """각 손가락의 굽힘 정도 계산 (0.0=굽힘, 1.0=펴짐)"""
        finger_states = {}
        
        for finger_name, indices in self.finger_indices.items():
            if finger_name == 'thumb':
                # 엄지: 각도 기반 계산
                p1 = np.array([landmarks_px[indices[0]]['x'], landmarks_px[indices[0]]['y']])
                p2 = np.array([landmarks_px[indices[1]]['x'], landmarks_px[indices[1]]['y']]) 
                p3 = np.array([landmarks_px[indices[3]]['x'], landmarks_px[indices[3]]['y']])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                normalized = angle / 180.0
                
            else:
                # 나머지 손가락: 거리 기반 계산
                mcp = np.array([landmarks_px[indices[0]]['x'], landmarks_px[indices[0]]['y']])
                pip = np.array([landmarks_px[indices[1]]['x'], landmarks_px[indices[1]]['y']])
                tip = np.array([landmarks_px[indices[3]]['x'], landmarks_px[indices[3]]['y']])
                
                direct_dist = np.linalg.norm(tip - mcp)
                joint_dist = np.linalg.norm(pip - mcp) + np.linalg.norm(tip - pip)
                
                ratio = direct_dist / (joint_dist + 1e-6)
                normalized = max(0.0, min(1.0, ratio))
            
            finger_states[finger_name] = normalized
        
        return finger_states
    
    def get_statistics(self) -> Dict:
        """추적 통계 반환"""
        detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
        return {
            'total_frames': self.frame_count,
            'detected_frames': self.detection_count,
            'detection_rate': detection_rate,
            'timestamp_ms': self.timestamp_ms
        }
    
    def close(self):
        """리소스 정리 (중요: 메모리 누수 방지)"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
            print("✓ HandLandmarker resources released")


# 독립 실행 테스트
if __name__ == "__main__":
    print("="*60)
    print("HAND TRACKER TEST (Tasks API)")
    print("="*60)
    
    # 모델 파일 확인
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print(f"\n❌ Model file not found: {HAND_LANDMARKER_MODEL_PATH}")
        print("\n📥 Download the model:")
        print("wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        exit(1)
    
    print(f"✓ Model file found: {HAND_LANDMARKER_MODEL_PATH}")
    print("\nPress 'Q' to quit")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)
    
    try:
        tracker = HandTracker()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 손 추적
            hand_data = tracker.process(frame)
            
            # 결과 표시
            if hand_data:
                cv2.putText(frame, "HAND DETECTED! (Tasks API)", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 손가락 상태 표시
                y = 90
                for finger, value in hand_data['finger_states'].items():
                    text = f"{finger}: {value:.2f}"
                    color = (0, 255, 0) if value > 0.5 else (0, 165, 255)
                    cv2.putText(frame, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y += 30
            else:
                cv2.putText(frame, "NO HAND - Show hand to camera", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Hand Tracker Test (Tasks API)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'tracker' in locals():
            tracker.close()
        cap.release()
        cv2.destroyAllWindows()
