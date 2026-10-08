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