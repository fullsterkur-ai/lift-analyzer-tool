import numpy as np

from enum import Enum
from typing import Optional, Union
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from . import EstimationStrategy

class PoseEstimationStrategies(Enum):
    MEDIAPIPE = 1

class EstimationStrategyManager:

    def __init__(self):
        self.strategies = {}
        self.active_strategy_name: Optional[PoseEstimationStrategies] = None
        self.active_strategy: Optional[EstimationStrategy] = None

    def register_strategy(self, strategy_name: PoseEstimationStrategies, strategy: EstimationStrategy):
        self.strategies[strategy_name] = strategy

    def set_strategy(self, strategy: PoseEstimationStrategies):
        self.active_strategy_name = strategy
        new_strategy = self.strategies[strategy]
        if new_strategy: self.active_strategy = new_strategy()

    def remove_strategy(self):
        self.active_strategy = None

    def estimate(self, frame: np.ndarray, timestamp: int):
        if not self.active_strategy:
            return frame, None
        return self.active_strategy.estimate(frame, timestamp)
    
    def draw_estimation(self, frame: np.ndarray, estimations: Union[PoseLandmarkerResult]):
        if not self.active_strategy:
            return frame
        return self.active_strategy.draw_estimations(frame, estimations)
 