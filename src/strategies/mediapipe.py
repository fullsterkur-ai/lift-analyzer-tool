import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import tkinter as tk

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, PoseLandmarker, \
    RunningMode, drawing_utils, drawing_styles, PoseLandmarksConnections

from . import EstimationStrategy

# credits for guide
# https://colab.sandbox.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb

class MediaPipeEstimationStrategy(EstimationStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_path = str(os.getcwd() / Path("assets/models/pose_landmarker_heavy.task"))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO
        
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.fig = None
        self.ax = None
        self.canvas = None


    def estimate(self, frame, timestamp):
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        return frame, self.landmarker.detect_for_video(img, timestamp)
    

    def _init_3d_plot(self):
        window = tk.Toplevel(tk._get_default_root())
        window.title("3D Pose")

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.view_init(elev=-90, azim=-90)

    def plot_world_landmarks(self, estimations):
        landmarks = estimations.pose_world_landmarks
        if not landmarks:
            return

        if self.fig is None:
            self._init_3d_plot()

        self.ax.cla()

        for pose_landmarks in landmarks:
            xs = [lm.x for lm in pose_landmarks]
            ys = [lm.y for lm in pose_landmarks]
            zs = [lm.z for lm in pose_landmarks]
            self.ax.scatter(xs, ys, zs)

            for connection in PoseLandmarksConnections.POSE_LANDMARKS:
                start, end = connection.start, connection.end
                self.ax.plot(
                    [xs[start], xs[end]],
                    [ys[start], ys[end]],
                    [zs[start], zs[end]],
                    'g-'
                )

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.canvas.draw()

    def plot_landmarks(self, frame, estimations):
        landmarks = estimations
        pose_landmarks_list = landmarks.pose_landmarks
        annotated_image = frame.copy()

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
          drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=pose_landmarks,
              connections=PoseLandmarksConnections.POSE_LANDMARKS,
              landmark_drawing_spec=pose_landmark_style,
              connection_drawing_spec=pose_connection_style)

        return annotated_image

    def draw_estimations(self, frame, estimations):

        self.plot_world_landmarks(estimations)
        return self.plot_landmarks(frame, estimations)

    