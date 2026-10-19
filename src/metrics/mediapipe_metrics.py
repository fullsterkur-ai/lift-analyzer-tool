import statistics
import cv2
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from .base_metrics import BaseMetrics
from ..utils.math import get_center, get_velocity, remove_jitters, smooth
from ..utils.biomechanics import calculate_hip_angle, calculate_knee_angle, \
    calculate_ankle_angle, calculate_elbow_angle, calculate_shoulder_angle, calculate_wrist_angle, \
    calculate_phases

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
        self.ax_y = self.fig.add_subplot(221)
        self.ax_vy = self.fig.add_subplot(222, sharex=self.ax_y)
        self.ax_theta = self.fig.add_subplot(223)
        self.ax_vtheta = self.fig.add_subplot(224)
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
        if len(estimations.pose_landmarks)  == 0:
            return (0, 0, 0)
        landmarks = estimations.pose_landmarks[0]
        left_index_finger, right_index_finger = landmarks[19], landmarks[20]
        return get_center(left_index_finger, right_index_finger)

    def get_3d_center(self, frame: np.ndarray, estimations: PoseLandmarkerResult):
        if len(estimations.pose_landmarks) == 0:
            return (0, 0, 0)
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

        self.history["angular_distances"]["hip"].append(calculate_hip_angle(estimations))
        self.history["angular_distances"]["knee"].append(calculate_knee_angle(estimations))
        self.history["angular_distances"]["ankle"].append(calculate_ankle_angle(estimations))
        self.history["angular_distances"]["wrist"].append(calculate_wrist_angle(estimations))
        self.history["angular_distances"]["elbow"].append(calculate_elbow_angle(estimations))
        self.history["angular_distances"]["shoulder"].append(calculate_shoulder_angle(estimations))


        ts = np.asarray(self.history["timestamps"], dtype=float)
        t = (ts - ts[0]) / 1_000_000_000.0

        pos = np.asarray(self.history["displacement"])

        history = {
            "time": t,
            "displacement": remove_jitters(smooth(pos[:, 1] - pos[0, 1])),
            "angular_distances": {
                "hip": remove_jitters(smooth(self.history["angular_distances"]["hip"])),
                "knee": remove_jitters(smooth(self.history["angular_distances"]["knee"])),
                # "ankle": remove_jitters(smooth(self.history["angular_distances"]["ankle"])),
                # "wrist": remove_jitters(smooth(self.history["angular_distances"]["wrist"])),
                "elbow": remove_jitters(smooth(self.history["angular_distances"]["elbow"])),
                "shoulder": remove_jitters(smooth(self.history["angular_distances"]["shoulder"])),
            }
        }
        history["velocity"] = get_velocity(history["displacement"], t)
        history["angular_velocities"] = {
            "hip": get_velocity(history["angular_distances"]["hip"], t),
            "knee": get_velocity(history["angular_distances"]["knee"], t),
            # "ankle": get_velocity(history["angular_distances"]["ankle"], t),
            # "wrist": get_velocity(history["angular_distances"]["wrist"], t),
            "elbow": get_velocity(history["angular_distances"]["elbow"], t),
            "shoulder": get_velocity(history["angular_distances"]["shoulder"], t),
        }

        return history

    def _clear_plots(self):
        self.ax_y.cla()
        self.ax_vy.cla()
        self.ax_theta.cla()
        self.ax_vtheta.cla()

    def plot_metrics(self, metrics):
        if self.fig is None:
            self._init_metrics_plot()
        
        if len(metrics["time"]) < 5:
            return

        t = metrics["time"]

        self._clear_plots()
        self.ax_y.plot(t, -metrics["displacement"])
        self.ax_y.set_ylabel("Displacement (m)")

        self.ax_vy.plot(t, -metrics["velocity"])
        self.ax_vy.axhline(0, color="gray", linewidth=0.5)
        self.ax_vy.set_ylabel("Velocity (m/s)")
        self.ax_vy.set_xlabel("Time (s)")

        self.ax_theta.plot(t, metrics["angular_distances"]["hip"], label="Hip")
        self.ax_theta.plot(t, metrics["angular_distances"]["knee"], label="Knee")
        # self.ax_theta.plot(t, metrics["angular_distances"]["ankle"], label="Ankle")
        # self.ax_theta.plot(t, metrics["angular_distances"]["wrist"], label="Wrist")
        self.ax_theta.plot(t, metrics["angular_distances"]["elbow"], label="Elbow")
        self.ax_theta.plot(t, metrics["angular_distances"]["shoulder"], label="Shoulder")
        self.ax_theta.legend()
        self.ax_theta.set_ylabel("Angle (deg)")
        self.ax_theta.set_xlabel("Time (s)")

        self.ax_vtheta.plot(t, metrics["angular_velocities"]["hip"], label="Hip")
        self.ax_vtheta.plot(t, metrics["angular_velocities"]["knee"], label="Knee")
        # self.ax_vtheta.plot(t, metrics["angular_velocities"]["ankle"], label="Ankle")
        # self.ax_vtheta.plot(t, metrics["angular_velocities"]["wrist"], label="Wrist")
        self.ax_vtheta.plot(t, metrics["angular_velocities"]["elbow"], label="Elbow")
        self.ax_vtheta.plot(t, metrics["angular_velocities"]["shoulder"], label="Shoulder")
        self.ax_vtheta.axhline(0, color="gray", linewidth=0.5)
        self.ax_vtheta.legend()
        self.ax_vtheta.set_ylabel("Angular velocity (deg/s)")
        self.ax_vtheta.set_xlabel("Time (s)")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def reset_metrics(self):
        self.history = {
            "displacement": [],
            "angular_distances": {
                "hip": [],
                "knee": [],
                "ankle": [],
                "wrist": [],
                "elbow": [],
                "shoulder": [],
            },
            "timestamps": [],
            "velocity": [],
            "angular_velocities": {
                "hip": [],
                "knee": [],
                "ankle": [],
                "wrist": [],
                "elbow": [],
                "shoulder": [],
            }
        }

    def get_current_lifting_phase(self, metrics, exercise):
        return calculate_phases(exercise, metrics)[-1]

    def draw_current_lifting_phase(self, frame, phase):
        colors = {
            "other": (200, 200, 200),
            "concentric": (80, 255, 80),
            "eccentric": (80, 80, 255),
            "static": (255, 170, 60),
        }
        cv2.putText(frame, phase, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[phase], 2)

    def draw_reps(self, frame, metrics, rep_ranges):
        cv2.putText(frame, "reps: " + str(len(rep_ranges)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        t = metrics["time"]
        xlen = len(t)
        for repr in rep_ranges:
            start, stop = repr
            tstart = t[start]
            tstop = t[-1] if stop >= xlen else t[stop]
            self.ax_y.axvspan(tstart, tstop, color="blue", alpha=0.15)
            self.ax_vy.axvspan(tstart, tstop, color="blue", alpha=0.15)
            self.ax_theta.axvspan(tstart, tstop, color="blue", alpha=0.15)
            self.ax_vtheta.axvspan(tstart, tstop, color="blue", alpha=0.15)

    def get_average_velocity(self, metrics, rep_ranges):
        v = -metrics["velocity"]
        if len(v) == 0 or len(rep_ranges) == 0:
            return 0
        velocities_per_rep = [statistics.mean(v[repr[0]:repr[1]]) for repr in rep_ranges]
        # return the instance with highest velocity
        return max(velocities_per_rep)

    def draw_average_velocity(self, frame, metrics, rep_ranges):
        avg_velocity = self.get_average_velocity(metrics, rep_ranges)
        cv2.putText(frame, f"avg velocity: {avg_velocity:.2f} m/s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


    def get_peak_velocity(self, metrics, rep_ranges):
        v = -metrics["velocity"]
        if len(v) == 0 or len(rep_ranges) == 0:
            return 0
        velocities_per_rep = [max(v[repr[0]:repr[1]]) for repr in rep_ranges]
        # return the instance with highest velocity
        return max(velocities_per_rep)


    def draw_peak_velocity(self, frame, metrics, rep_ranges):
        peak_velocity = self.get_peak_velocity(metrics, rep_ranges)
        cv2.putText(frame, f"peak velocity: {peak_velocity:.2f} m/s", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

