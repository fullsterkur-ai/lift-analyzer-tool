import numpy as np

from abc import ABC, abstractmethod

class BaseMetrics(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_2d_center(self, frame: np.ndarray, estimations, *args, **kwargs):
        ...

    @abstractmethod
    def get_3d_center(self, frame: np.ndarray, estimations, *args, **kwargs):
        ...

    @abstractmethod
    def draw_2d_center(self, frame: np.ndarray, estimations, *args, **kwargs):
        ...

    @abstractmethod
    def draw_3d_center(self, frame: np.ndarray, estimations, *args, **kwargs):
        ...

    @abstractmethod
    def get_metrics(self, frame: np.ndarray, estimations, *args, **kwargs):
        ...

    @abstractmethod
    def plot_metrics(self, *args, **kwargs):
        ...

    @abstractmethod
    def reset_metrics(self):
        ...
    
    @abstractmethod
    def get_current_lifting_phase(self, metrics, exercise: str, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def draw_current_lifting_phase(self, frame: np.ndarray, phase: str, *args, **kwargs):
        ...
    
    @abstractmethod
    def draw_reps(self, frame: np.ndarray, metrics, rep_ranges):
        ...

    @abstractmethod
    def get_average_velocity(self, metrics, rep_ranges):
        ...

    @abstractmethod
    def get_peak_velocity(self, metrics, rep_ranges):
        ...

    @abstractmethod
    def draw_average_velocity(self, frame: np.ndarray, metrics, rep_ranges):
        ...

    @abstractmethod
    def draw_peak_velocity(self, frame: np.ndarray, metrics, rep_ranges):
        ...

    @abstractmethod
    def get_rpe(self, metrics, rep_ranges):
        ...

    @abstractmethod
    def draw_rpe(self, frame: np.ndarray, metrics, rep_ranges):
        ...
