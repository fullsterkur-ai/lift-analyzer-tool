import re
import cv2
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from .videosource import VideoSource
from .strategies.mediapipe import MediaPipeEstimationStrategy
from .metrics.mediapipe_metrics import MediapipeMetrics
from .estimations.max_daily import MaxDailyLinearRegressionModel, MaxDailyLogisticRegressionModel,\
    predict_1rm_with_mvt, mvt, exclude_coldstarts
from .utils.biomechanics import get_rep_ranges

EXERCISES = [
    "Squat",
    "Deadlift",
    "Bench Press",
    "Overhead Press",
    "Barbell Row",
    "Pull-up",
    "Romanian Deadlift",
    "Front Squat",
    "Incline Bench Press",
    "Dip",
]


class SetRow:

    def __init__(self, parent, number: int, on_remove):
        self.video_path: str | None = None
        self._thumb = None  # keep a reference so it isn't garbage collected
        self._on_remove = on_remove

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=3)

        self.number_label = ttk.Label(self.frame, text=f"#{number}", width=4)
        self.number_label.pack(side="left")

        self.weight_var = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.weight_var, width=8).pack(side="left")
        ttk.Label(self.frame, text="kg").pack(side="left", padx=(2, 10))

        ttk.Button(
            self.frame, text="Upload footage", command=self._browse
        ).pack(side="left")

        self.thumb_label = ttk.Label(self.frame)
        self.thumb_label.pack(side="left", padx=8)

        self.name_label = ttk.Label(self.frame, text="No file")
        self.name_label.pack(side="left")

        ttk.Button(
            self.frame, text="✕", width=3, command=lambda: self._on_remove(self)
        ).pack(side="right")

    def set_number(self, number: int):
        self.number_label.config(text=f"#{number}")

    def destroy(self):
        self.frame.destroy()

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select footage",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.set_footage(path)
        
    def set_footage(self, path: str):
        self.video_path = path
        self.name_label.config(text=path.replace("\\", "/").split("/")[-1])
        self._show_thumbnail(path)
        
        if not self.weight_var.get():
            match = re.search(r"\/.*\D+(\d+)kg.*\..+$", path, re.IGNORECASE)
            if match:
                self.weight_var.set(match.group(1))

    def _show_thumbnail(self, path: str):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.thumb_label.config(image="", text="(no preview)")
            return
        frame = cv2.resize(frame, (32, 24))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._thumb = ImageTk.PhotoImage(img)
        self.thumb_label.config(image=self._thumb, text="")

    def get_data(self) -> dict:
        try:
            weight = float(self.weight_var.get())
        except ValueError:
            weight = None
        return {"weight_kg": weight, "video_path": self.video_path}


class ScheduleForm(tk.Toplevel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Schedule")
        self.geometry("560x400")

        self.rows: list[SetRow] = []

        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Exercise:").pack(side="left")
        self.exercise_var = tk.StringVar(value=EXERCISES[0])
        ttk.Combobox(
            top, textvariable=self.exercise_var, values=EXERCISES, state="readonly"
        ).pack(side="left", padx=5)

        self.rows_frame = ttk.Frame(main)
        self.rows_frame.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(anchor="w", fill="both")
        ttk.Button(btn_frame, text="Add set", command=self.add_set).pack(padx=5, side="left")
        ttk.Button(btn_frame, text="Bulk upload footage", command=self.bulk_upload_footage).pack(padx=5, side="left")
        ttk.Button(btn_frame, text="Process", command=self.process).pack(anchor="e", padx=5, side="right")

        self.add_set()

    def add_set(self):
        row = SetRow(self.rows_frame, len(self.rows) + 1, on_remove=self.remove_set)
        self.rows.append(row)
        return row

    def bulk_upload_footage(self):
        paths = filedialog.askopenfilenames(
            title="Select footage",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        for path in paths:
            row = self.add_set()
            row.set_footage(path)

    def remove_set(self, row: SetRow):
        row.destroy()
        self.rows.remove(row)
        for i, r in enumerate(self.rows, start=1):
            r.set_number(i)

    def get_data(self) -> dict:
        return {
            "exercise": self.exercise_var.get(),
            "sets": [row.get_data() for row in self.rows],
        }
    
        
    def _process_video(self, path):
        source = VideoSource.from_filepicker(path)
        source.capture_video()

        metrics = MediapipeMetrics()
        estimation_strat = MediaPipeEstimationStrategy()

        source.reset_playback()
        metrics.reset_metrics()

        estimated = []
        while not source.playback_is_empty():
            frame, timestamp, _ = source.next()
            frame, estimations = estimation_strat.estimate(frame, timestamp // 1_000_000)
            estimated.append((frame, estimations, timestamp))

        metrics.calibrate_scale([(f, e) for f, e, _ in estimated])

        history = None
        for frame, estimations, timestamp in estimated:
            history = metrics.get_metrics(frame, estimations, timestamp)

        exercise = self.exercise_var.get().lower()
        rep_ranges = get_rep_ranges(exercise, history)
        return {
            "average_velocity": metrics.get_average_velocity(history, rep_ranges),
            "peak_velocity": metrics.get_peak_velocity(history, rep_ranges)
        }

    def process(self):
        average_velocities = []
        peak_velocities = []
        exercise = self.exercise_var.get()

        weights = list(map(lambda row: float(row.weight_var.get()), self.rows))
        for rep_row in self.rows:
            processed = self._process_video(rep_row.video_path)
            average_velocities.append(processed["average_velocity"])
            peak_velocities.append(processed["peak_velocity"])

        plt.cla()
        plt.plot(weights, average_velocities, "x-", label="Average velocity")
        plt.plot(weights, peak_velocities, "x-", label="Peak velocity")
        
        weights, average_velocities = exclude_coldstarts(weights, average_velocities)
        _, peak_velocities = exclude_coldstarts(weights, peak_velocities)

        X = np.linspace(min(weights), max(weights) * 1.5, 100)
        avg_vel_linear_reg = MaxDailyLinearRegressionModel()
        avg_vel_linear_reg.fit(weights, average_velocities, 2)
        plt.plot(X, avg_vel_linear_reg.predict(X), label="Predicted average velocities (linear)")

        peak_vel_linear_reg = MaxDailyLinearRegressionModel()
        peak_vel_linear_reg.fit(weights, peak_velocities, 2)
        plt.plot(X, peak_vel_linear_reg.predict(X), label="Predicted peak velocities (linear)")

        one_rep_max = max(
            predict_1rm_with_mvt(exercise, avg_vel_linear_reg.coeffs) or 0,
            *weights
        )
        
        failure_line = mvt(exercise) or 0.05
        plt.axhline(failure_line, color="gray", linestyle=":")
        plt.text(one_rep_max, failure_line, f"1RM: {one_rep_max:.2f}kg")

        plt.xlabel("Weight (kg)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.show()
