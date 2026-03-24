#hand_tracker.py
"""
MediaPipe를 사용한 핸드 트래킹 및 특징 추출
"""
import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Dict, List
import logging

from config.constants import FINGER_ORDER, FINGER_LANDMARKS

logger = logging.getLogger(__name__)


class HandTrackerTasks:
    """핸드 랜드마크 감지 및 특징 추출 클래스"""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        logger.info("핸드 트래커 초기화 완료")

    def process(self, frame_bgr: np.ndarray) -> Tuple[
        np.ndarray, Optional[Dict], Optional[Dict], Optional[Tuple], Optional[List]]:
        """
        프레임 처리 및 손 특징 추출

        Returns:
            Tuple of (annotated_frame, curls, splay, thumb_pair, landmarks_xy)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.detector.detect(mp_img)
        except Exception as e:
            logger.error(f"핸드 트래킹 처리 실패: {e}")
            return frame_bgr, None, None, None, None

        if not result.hand_landmarks:
            return frame_bgr, None, None, None, None

        # 왼손만 필터링 (프레임이 플립되어 있어서 "Right"가 실제 왼손)
        hand_idx = None
        if hasattr(result, "handedness") and result.handedness:
            for idx, handed_list in enumerate(result.handedness):
                if handed_list and len(handed_list) > 0:
                    label = handed_list[0].category_name
                    if label == "Right":
                        hand_idx = idx
                        break

        if hand_idx is None:
            return frame_bgr, None, None, None, None

        landmarks = result.hand_landmarks[hand_idx]
        landmarks_np = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        landmarks_xy = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        self._draw_landmarks(frame_bgr, landmarks_xy)

        # 특징 계산
        curls = {f: self._calculate_finger_curl(landmarks_np, f) for f in FINGER_ORDER}
        splay = self._compute_splay_degrees(landmarks_np)
        thumb_mcp_curl, thumb_ip_curl = self._calculate_thumb_curls(landmarks_np)

        # 엄지 대립(Opposition) 계산 추가
        thumb_opposition = self._calculate_thumb_opposition(landmarks_np)

        # 3개 값으로 반환
        return frame_bgr, curls, splay, (thumb_mcp_curl, thumb_ip_curl, thumb_opposition), landmarks_xy

    def _calculate_thumb_opposition(self, landmarks: np.ndarray) -> float:
        """
        엄지 대립(Opposition) 정도 계산
        엄지 끝과 새끼 기저부 사이의 거리를 기반으로 계산

        Returns:
            opposition 값 (0.0: 완전 펼침, 1.0: 완전 대립)
        """
        thumb_tip = landmarks[4]  # 엄지 끝
        pinky_mcp = landmarks[17]  # 새끼 MCP
        wrist = landmarks[0]  # 손목

        # 정규화를 위한 기준 거리 (손목-새끼MCP)
        reference_distance = np.linalg.norm(pinky_mcp - wrist)
        if reference_distance < 1e-9:
            return 0.0

        # 엄지 끝과 새끼 MCP 사이의 거리
        thumb_pinky_distance = np.linalg.norm(thumb_tip - pinky_mcp)

        # 정규화된 거리
        normalized_distance = thumb_pinky_distance / reference_distance

        # 거리 기반 Opposition 계산 (실험적 값, 튜닝 필요)
        max_distance = 1.2  # 완전히 펼쳤을 때
        min_distance = 0.3  # 완전히 붙였을 때

        # 거리가 가까울수록 Opposition 값이 높아짐
        opposition = 1.0 - np.clip(
            (normalized_distance - min_distance) / (max_distance - min_distance),
            0.0, 1.0
        )

        # 디버깅용 출력 (초기 테스트 시)
        # print(f"Opposition: {opposition:.3f}")

        return float(opposition)

    @staticmethod
    def _calculate_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 간의 각도를 도 단위로 계산"""
        dot_product = float(np.dot(v1, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 < 1e-9 or norm2 < 1e-9:
            return 180.0

        cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def _curl_from_joint_angle(angle_deg: float, open_deg: float = 170.0, closed_deg: float = 70.0) -> float:
        """관절 각도를 컬 값 [0,1]로 변환"""
        curl = (open_deg - angle_deg) / (open_deg - closed_deg)
        return float(np.clip(curl, 0.0, 1.0))

    def _calculate_finger_curl(self, landmarks: np.ndarray, finger: str) -> float:
        """손가락 컬 값 계산"""
        indices = FINGER_LANDMARKS[finger]
        points = [landmarks[i] for i in indices]

        if finger == "finger1":  # 엄지
            wrist = landmarks[0]
            tip = landmarks[4]
            reference = landmarks[2]
            current_distance = np.linalg.norm(tip - wrist)
            max_distance = np.linalg.norm(reference - wrist) * 1.4

            if max_distance < 1e-9:
                return 0.0

            curl = 1.0 - min(current_distance / max_distance, 1.0)
            return float(np.clip(curl, 0.0, 1.0))

        # 다른 손가락들
        mcp, pip, dip, tip = points
        pip_angle = self._calculate_angle_degrees(mcp - pip, dip - pip)
        dip_angle = self._calculate_angle_degrees(pip - dip, tip - dip)
        average_angle = 0.6 * pip_angle + 0.4 * dip_angle

        return self._curl_from_joint_angle(average_angle)

    def _calculate_thumb_curls(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """엄지 MCP 및 IP 컬 값 계산"""
        p1, p2, p3, p4 = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
        mcp_angle = self._calculate_angle_degrees(p1 - p2, p3 - p2)
        ip_angle = self._calculate_angle_degrees(p2 - p3, p4 - p3)

        return (self._curl_from_joint_angle(mcp_angle),
                self._curl_from_joint_angle(ip_angle))

    @staticmethod
    def _normalize_2d_vector(v: np.ndarray) -> np.ndarray:
        """2D 벡터 정규화"""
        norm = float(np.linalg.norm(v))
        return v / (norm + 1e-9)

    @staticmethod
    def _signed_angle_2d(a: np.ndarray, b: np.ndarray) -> float:
        """2D 벡터 간의 부호 있는 각도 계산"""
        a_norm = HandTrackerTasks._normalize_2d_vector(a)
        b_norm = HandTrackerTasks._normalize_2d_vector(b)
        return math.degrees(math.atan2(a_norm[0] * b_norm[1] - a_norm[1] * b_norm[0],
                                       a_norm[0] * b_norm[0] + a_norm[1] * b_norm[1]))

    @staticmethod
    def _finger_direction_2d(landmarks: np.ndarray, mcp_idx: int, tip_idx: int) -> np.ndarray:
        """2D 손가락 방향 벡터 계산"""
        direction = landmarks[tip_idx] - landmarks[mcp_idx]
        return np.array([direction[0], direction[1]], dtype=np.float32)

    def _compute_splay_degrees(self, landmarks: np.ndarray) -> Dict[str, float]:
        """중지를 기준으로 한 손가락 벌림 각도 계산"""
        directions = {
            "finger2": self._finger_direction_2d(landmarks, 5, 8),
            "finger3": self._finger_direction_2d(landmarks, 9, 12),
            "finger4": self._finger_direction_2d(landmarks, 13, 16),
            "finger5": self._finger_direction_2d(landmarks, 17, 20),
            "finger1": self._finger_direction_2d(landmarks, 2, 4),
        }

        base_direction = directions["finger3"]

        return {
            "finger3": 0.0,
            "finger2": self._signed_angle_2d(base_direction, directions["finger2"]),
            "finger4": self._signed_angle_2d(base_direction, directions["finger4"]),
            "finger5": self._signed_angle_2d(base_direction, directions["finger5"]),
            "finger1": self._signed_angle_2d(base_direction, directions["finger1"]),
        }

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, landmarks_xy: List[Tuple[int, int]]):
        """프레임에 핸드 랜드마크 및 연결선 그리기"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # 연결선 그리기
        for start, end in connections:
            cv2.line(frame, landmarks_xy[start], landmarks_xy[end], (0, 255, 0), 2)

        # 랜드마크 점 그리기
        for i, (x, y) in enumerate(landmarks_xy):
            color, radius = (0, 0, 255), 4
            if i == 0:  # 손목
                color, radius = (255, 0, 0), 7
            elif i in [4, 8, 12, 16, 20]:  # 손가락 끝
                color, radius = (0, 255, 255), 6
            cv2.circle(frame, (x, y), radius, color, -1)
