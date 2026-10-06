import cv2
import numpy as np

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from .base_metrics import BaseMetrics
from ..utils.math import get_center

class MediapipeMetrics(BaseMetrics):

    def __init__(self):
        super().__init__()

    def get_2d_center(self, frame: np.ndarray, estimations: PoseLandmarkerResult):
        landmarks = estimations.pose_landmarks[0]
        left_index_finger, right_index_finger = landmarks[19], landmarks[20]
        return get_center(left_index_finger, right_index_finger)

    def get_3d_center(self, frame: np.ndarray, estimations: PoseLandmarkerResult):
        landmarks = estimations.pose_world_landmarks[0]
        left_index_finger, right_index_finger = landmarks[19], landmarks[20]
        return get_center(left_index_finger, right_index_finger)

    def draw_2d_center(self, frame: np.ndarray, estimations: PoseLandmarkerResult):
        center = self.get_2d_center(frame, estimations)
        h, w = frame.shape[:2]
        cx, cy = int(center[0] * w), int(center[1] * h)

        cv2.circle(frame, (cx, cy), radius=8, color=(255, 0, 0), thickness=-1)
        return frame

    def draw_3d_center(self, frame: np.ndarray, estimations: PoseLandmarkerResult, ax):
        center = self.get_3d_center(frame, estimations)    
        ax.scatter(*center)