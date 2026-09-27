import cv2
import math
import tkinter as tk
import numpy as np
import threading

from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from .videosource import VideoSource, VideoCaptureListeners
from .strategies import EstimationStrategyManager, PoseEstimationStrategies

class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (w, h))

        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        self.video_frame = ttk.Frame(self.main_frame, relief="groove", borderwidth=3)
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)

        self.ctrl_frame = ttk.Frame(self.main_frame, relief="groove", borderwidth=3, padding=5)
        self.ctrl_frame.pack(side="left", fill="both")

        self.record_video_button = ttk.Button(
            self.ctrl_frame, text="Record video", command=self._handle_record_video
        )
        self.record_video_button.pack(fill="both", pady=1)

        self.upload_video_button = ttk.Button(
            self.ctrl_frame, text="Browse video from device", command=self._handle_upload_video
        )
        self.upload_video_button.pack(fill="both", pady=1)

        ttk.Separator(self.ctrl_frame).pack(pady=5, fill="x")

        self.play_pause_button = ttk.Button(
            self.ctrl_frame, text="Play", command=self._handle_play_pause, state="disabled"
        )
        self.play_pause_button.pack(fill="both", pady=1)

        ttk.Label(self.ctrl_frame, text="Strategies").pack(pady=5, fill="x")

        self.strategy_mediapipe_button = ttk.Button(
            self.ctrl_frame, text="Estimate with MediaPipe", command=self._set_mediapipe_strategy, state="disabled" 
        )
        self.strategy_mediapipe_button.pack(fill="both", pady=1)

        self.video_source: VideoSource | None = None
        self._capture_thread: threading.Thread | None = None

        self._is_recording = False
        self._is_playing = False
        self._play_after_id: str | None = None
        self._pose_strategy_manager = None

    def register_pose_strategy_manager(self, manager: EstimationStrategyManager):
        self._pose_strategy_manager = manager

    def _on_video_capture_frame(self, frame: np.ndarray):
        self.after(0, self._render_frame, frame)

    def _render_frame(self, frame: np.ndarray):
        frame_h, frame_w = frame.shape[:2]
        window_w, window_h = (self.video_frame.winfo_width(), self.video_frame.winfo_height()) 
        scale = min(window_w / frame_w, window_h / frame_h)
        new_w, new_h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))

        frame = cv2.resize(frame, (new_w, new_h))
        # apply padding
        if new_w < window_w:
            pad_w = math.floor((window_w - new_w) / 2)
            padding = np.zeros((frame.shape[0], pad_w, frame.shape[2]), dtype=frame.dtype)
            frame = np.hstack((padding, frame, padding))
        if new_h < window_h:
            pad_h = math.floor((window_h - new_h) / 2)
            padding = np.zeros((pad_h, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
            frame = np.vstack((padding, frame, padding))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage collected
        self.video_label.configure(image=imgtk)

    def _start_capture(self, video_src: VideoSource):
        self.video_source = video_src
        listeners = VideoCaptureListeners(
            on_frame=self._on_video_capture_frame,
            on_end=self._on_capture_end,
        )
        self._capture_thread = threading.Thread(
            target=video_src.capture_video,
            kwargs={"listeners": listeners},
            daemon=True,
        )
        self._capture_thread.start()

    def _on_capture_end(self):
        self.after(0, self._on_capture_end_main_thread)

    def _on_capture_end_main_thread(self):
        self._is_recording = False
        self.record_video_button.config(text="Record video")
        self.upload_video_button.config(state="normal")
        if self.video_source and self.video_source.buffer:
            self.play_pause_button.config(state="normal")
            self.strategy_mediapipe_button.config(state="normal")

    def _handle_record_video(self):
        if not self._is_recording:
            video_src = VideoSource.from_camera()
            self._is_recording = True
            self.record_video_button.config(text="Stop recording")
            self.upload_video_button.config(state="disabled")
            self.play_pause_button.config(state="disabled")
            self.strategy_mediapipe_button.config(state="disabled")
            self._start_capture(video_src)
        else:
            # stop button pressed mid-recording
            if self.video_source:
                self.video_source.stop()

    def _handle_upload_video(self):
        filepath = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            video_src = VideoSource.from_filepicker(filepath)
            self._is_recording = True
            self.record_video_button.config(text="Stop recording")
            self.upload_video_button.config(state="disabled")
            self.play_pause_button.config(state="disabled")
            self._start_capture(video_src)

    def _handle_play_pause(self):
        # remove any estimation strategy as we don't want that
        self._pose_strategy_manager.remove_strategy()
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self.video_source or not self.video_source.buffer:
            return
        self.video_source.reset_playback()
        self._is_playing = True
        self.play_pause_button.config(text="Stop")
        self.record_video_button.config(state="disabled")
        self.upload_video_button.config(state="disabled")
        self._play_next_frame()

    def _play_next_frame(self):
        if not self._is_playing:
            return
        if self.video_source.playback_is_empty():
            self._stop_playback()
            return
        
        frame, timestamp, next_timestamp = self.video_source.next()
        if self._pose_strategy_manager:
            frame, estimations = self._pose_strategy_manager.estimate(frame, timestamp // 1_000_000)
            frame = self._pose_strategy_manager.draw_estimation(frame, estimations)
        self._render_frame(frame)

        if self.video_source.source_fps:
            delay_ms = max(1, int(1000 / self.video_source.source_fps))
        else:
            delay_ms = max(1, min(int((next_timestamp - timestamp) / 1_000_000), 200))
        self._play_after_id = self.after(delay_ms, self._play_next_frame)

    def _stop_playback(self):
        self._is_playing = False
        self.play_pause_button.config(text="Play")
        self.record_video_button.config(state="normal")
        self.upload_video_button.config(state="normal")
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _set_mediapipe_strategy(self):
        self._pose_strategy_manager.set_strategy(PoseEstimationStrategies.MEDIAPIPE)
        self._start_playback()

    def destroy(self):
        self._stop_playback()
        if self.video_source:
            self.video_source.stop()
        super().destroy()
