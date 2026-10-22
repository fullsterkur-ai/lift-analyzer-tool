import numpy as np

from abc import ABC, abstractmethod
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from typing import Union

class EstimationStrategy(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def estimate(self, frame: np.ndarray, timestamp: int) -> tuple[np.ndarray, PoseLandmarkerResult]:
        ...

    @abstractmethod
    def draw_estimations(self, frame: np.ndarray, estimations: Union[PoseLandmarkerResult]) -> np.ndarray:
        ...