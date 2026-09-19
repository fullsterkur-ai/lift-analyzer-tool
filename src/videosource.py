import cv2
import time
import numpy as np

from typing import Optional, Callable
from os import PathLike
from cv2 import VideoCapture

class VideoCaptureListeners:

    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
    ):
        self.on_start = on_start
        self.on_frame = on_frame
        self.on_end = on_end

class VideoSource:

    def __init__(self, cap_source: str | PathLike[str] | int, target_dim: tuple[int, int]):
        self.buffer = []
        self.buffer_timestamps = []
        self.cap = VideoCapture(cap_source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {cap_source}")
        self.target_dim = target_dim
        self._stop_requested = False
        self.replay_idx = 0

    @staticmethod
    def from_filepicker(filepath, *args, **kwargs):
        return VideoSource(filepath, *args, **kwargs)

    @staticmethod
    def from_camera(*args, **kwargs):
        return VideoSource(0, *args, **kwargs)

    def stop(self):
        self._stop_requested = True

    def next(self, reset=False):
        if self.replay_idx < 0:
            self.replay_idx = 0
        curr_frame = self.buffer[self.replay_idx]
        curr_timestamp = self.buffer_timestamps[self.replay_idx]

        self.replay_idx += 1
        if self.playback_is_empty() and reset:
            self.replay_idx = 0

        next_idx = self.replay_idx if self.replay_idx < len(self.buffer) else 0
        next_timestamp = self.buffer_timestamps[next_idx]

        return curr_frame, curr_timestamp, next_timestamp
    
    def playback_is_empty(self):
        return self.replay_idx >= len(self.buffer)

    def capture_video(self, listeners: Optional[VideoCaptureListeners] = None):
        self._stop_requested = False
        if listeners and listeners.on_start: listeners.on_start()

        while not self._stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.buffer.append(frame)
            self.buffer_timestamps.append(time.monotonic_ns())
            frame = cv2.resize(frame, self.target_dim)
            if listeners and listeners.on_frame: listeners.on_frame(frame)

        self.cap.release()
        if listeners and listeners.on_end: listeners.on_end()
        return self.buffer
