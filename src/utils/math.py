import math
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

from .point3d import Point3D

def get_center(p1, p2):
    return (
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
        (p1.z + p2.z) / 2,
    )

# credits: https://stackoverflow.com/a/19730129
def get_angle(p1, p2, p3):
    v1 = Point3D(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
    v2 = Point3D(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z)

    v1mag = math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z)
    v1norm = Point3D(v1.x / v1mag, v1.y / v1mag, v1.z / v1mag)

    v2mag = math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z)
    v2norm = Point3D(v2.x / v2mag, v2.y / v2mag, v2.z / v2mag)

    res = v1norm.x * v2norm.x + v1norm.y * v2norm.y + v1norm.z * v2norm.z
    return math.degrees(math.acos(res))

def _odd_window(window, n):
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def smooth(values, window=11, poly=3):
    values = np.asarray(values, dtype=float)
    window = _odd_window(window, len(values))
    if window <= poly:
        return values
    return savgol_filter(values, window, poly, axis=0)

def remove_jitters(coordinates, median_kernel=5):
    return median_filter(coordinates, size=median_kernel, mode="nearest")

def get_velocity(y, t, window=31, poly=2):
    n = len(t)
    if n < 2:
        return np.zeros(n)

    window = _odd_window(window, n)
    if window <= poly:
        return np.gradient(y, t)

    dt = np.mean(np.diff(t))
    return savgol_filter(y, window, poly, deriv=1, delta=dt)