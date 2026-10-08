import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

def get_center(p1, p2):
    return (
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
        (p1.z + p2.z) / 2,
    )

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