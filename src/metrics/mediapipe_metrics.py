import cv2
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from .base_metrics import BaseMetrics
from ..utils.math import get_center, get_velocity, remove_jitters, smooth

class MediapipeMetrics(BaseMetrics):

    def __init__(self):
        super().__init__()
        self.history = {}
        self.reset_metrics()

        self.window = None
        self.fig = None
        self.ax = None
        self.canvas = None

    def _init_metrics_plot(self):
        self.window = tk.Toplevel(tk._get_default_root())
        self.window.title("Mediapipe metrics")

        self.fig = Figure(figsize=(6, 6))
        self.ax_y = self.fig.add_subplot(211)
        self.ax_vy = self.fig.add_subplot(212, sharex=self.ax_y)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close_metrics_window)

    def _on_close_metrics_window(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.window is not None:
            self.window.destroy()
        self.fig = None
        self.canvas = None
        self.window = None

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

    def get_metrics(self, frame: np.ndarray, estimations: PoseLandmarkerResult, timestamp):
        center = self.get_3d_center(frame, estimations)
        self.history["displacement"].append(center)
        self.history["timestamps"].append(timestamp)

        ts = np.asarray(self.history["timestamps"], dtype=float)
        t = (ts - ts[0]) / 1_000_000_000.0  # Convert nanoseconds to seconds

        pos = np.asarray(self.history["displacement"])
        y_raw = pos[:, 1] - pos[0, 1]
        y = remove_jitters(y_raw)

        return {
            "time": t,
            "displacement": smooth(y),
            "velocity": get_velocity(y, t),
        }

    def plot_metrics(self, metrics):
        if self.fig is None:
            self._init_metrics_plot()
        
        if len(metrics["time"]) < 5:
            return

        t = metrics["time"]
        displacement = -metrics["displacement"]
        velocity = -metrics["velocity"]

        self.ax_y.cla()
        self.ax_vy.cla()
        self.ax_y.plot(t, displacement)
        self.ax_y.set_ylabel("Displacement (m)")
        self.ax_vy.plot(t, velocity)
        self.ax_vy.axhline(0, color="gray", linewidth=0.5)
        self.ax_vy.set_ylabel("Velocity (m/s)")
        self.ax_vy.set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def reset_metrics(self):
        self.history = {
            "displacement": [],
            "timestamps": [],
            "velocity": []
        }
