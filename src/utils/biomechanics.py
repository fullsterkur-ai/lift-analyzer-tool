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

def calculate_body_angle(estimations: PoseLandmarkerResult, left_landmark_locations: list[Landmark]):
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
