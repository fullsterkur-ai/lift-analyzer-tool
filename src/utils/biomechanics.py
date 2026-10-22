import numpy as np
from enum import IntEnum
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from .math import get_angle

class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

PHASE_MAP = {
    "squat":                ("knee",     "increasing"),
    "leg extension":        ("knee",     "increasing"),
    "deadlift":             ("hip",      "increasing"),
    "romanian deadlift":    ("hip",      "increasing"),
    "hip thrust":           ("hip",      "increasing"),
    "leg raises":           ("hip",      "decreasing"),
    "push up":              ("elbow",    "increasing"),
    "bench press":          ("elbow",    "increasing"),
    "incline bench press":  ("elbow",    "increasing"),
    "decline bench press":  ("elbow",    "increasing"),
    "shoulder press":       ("elbow",    "increasing"),
    "tricep dips":          ("elbow",    "increasing"),
    "tricep pushdown":      ("elbow",    "increasing"),
    "barbell biceps curl":  ("elbow",    "decreasing"),
    "hammer curl":          ("elbow",    "decreasing"),
    "lat pulldown":         ("elbow",    "decreasing"),
    "pull up":              ("elbow",    "decreasing"),
    "t bar row":            ("elbow",    "decreasing"),
    "lateral raise":        ("shoulder", "increasing"),
}

OTHER, ECCENTRIC, CONCENTRIC, STATIC = 0, 1, 2, 3
PHASE_NAMES = ["other", "eccentric", "concentric", "static"]

STILL_FRAC = 0.15     # |velocity| below this fraction of the peak counts as "not moving"
LOCKOUT_ZONE = 0.15   # "at the lockout end" = within this fraction of the video's range of motion
MIN_LEN = 3           # runs shorter than this many frames get absorbed into the previous phase


def calculate_body_angle(estimations: PoseLandmarkerResult, left_landmark_locations: list[Landmark]):
    if len(estimations.pose_world_landmarks) == 0:
        return 0
    landmarks = estimations.pose_world_landmarks[0]
    if not landmarks:
        return 0
    
    left_angle = get_angle(
        landmarks[left_landmark_locations[0]],
        landmarks[left_landmark_locations[1]],
        landmarks[left_landmark_locations[2]]
    )

    right_angle = get_angle(
        landmarks[left_landmark_locations[0] + 1],
        landmarks[left_landmark_locations[1] + 1],
        landmarks[left_landmark_locations[2] + 1]
    )

    return (left_angle + right_angle) / 2

def calculate_hip_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP, Landmark.LEFT_KNEE ])

def calculate_knee_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_HEEL ])

def calculate_ankle_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_KNEE, Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX ])

def calculate_shoulder_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_HIP, Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW  ])

def calculate_elbow_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST ])

def calculate_wrist_angle(estimations: PoseLandmarkerResult):
    return calculate_body_angle(estimations, [ Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST, Landmark.LEFT_INDEX ])

def classify_phase(pos, vel, concentric):
    pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
    thr = STILL_FRAC * np.nanmax(np.abs(vel))
    rising, falling = vel > thr, vel < -thr
    still = ~(rising | falling)

    codes = np.full(len(vel), OTHER)
    if concentric == "increasing":
        codes[rising], codes[falling] = CONCENTRIC, ECCENTRIC
    else:
        codes[falling], codes[rising] = CONCENTRIC, ECCENTRIC

    # lockout = still AND sitting at the end of the concentric direction
    # (top of the angle range if the concentric part increases it, bottom if it decreases it)
    norm = (pos - np.nanmin(pos)) / (np.nanmax(pos) - np.nanmin(pos) + 1e-9)
    at_lockout = norm > 1 - LOCKOUT_ZONE if concentric == "increasing" else norm < LOCKOUT_ZONE
    codes[still & at_lockout] = STATIC

    # absorb very short runs into the previous phase
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            if i - start < MIN_LEN and start > 0:
                codes[start:i] = codes[start - 1]
            start = i
    return codes

def calculate_phases(exercise, metrics):
    # exercises not in PHASE_MAP fall back to vertical hand displacement (y points down, so up = decreasing)
    joint, concentric = PHASE_MAP.get(
        exercise.lower().replace("-", " ").replace("_", " "), (None, "decreasing")
    )

    pos = metrics["angular_distances"][joint] if joint else metrics["displacement"]
    vel = metrics["angular_velocities"][joint] if joint else metrics["velocity"]
    return [PHASE_NAMES[c] for c in classify_phase(pos, vel, concentric)]
     
def get_rep_ranges(exercise, metrics):
    phases = np.asarray(calculate_phases(exercise, metrics), dtype=str)
    # figure out continuous concentric phases
    # credits: https://stackoverflow.com/a/70396444
    concentric = (phases == "concentric").astype(np.int8)
    phase_diffs = np.diff(np.concatenate(([0], concentric, [0])))
    rep_starts, rep_ends = (np.flatnonzero(phase_diffs == 1), np.flatnonzero(phase_diffs == -1))
    return [(int(rep_starts[i]), int(rep_ends[i])) for i in range(len(rep_starts))]
