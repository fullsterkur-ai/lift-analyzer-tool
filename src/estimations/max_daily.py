import numpy as np
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression

class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

class MaxDailyLinearRegressionModel(Model):

    def __init__(self):
        super().__init__()
        self.p = None
        self.coeffs = None

    def fit(self, X, y, deg=1):
        self.coeffs = np.polyfit(X, y, deg)
        self.p = np.poly1d(self.coeffs)

    def predict(self, X):
        assert self.p is not None
        return self.p(X)

class MaxDailyLogisticRegressionModel(Model):

    def __init__(self):
        super().__init__()
        self.clf = None
        self.scale = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)

        self.scale = y.max()
        frac = y / self.scale

        X2 = np.vstack([X, X])
        labels = np.r_[np.ones(len(frac)), np.zeros(len(frac))]
        sample_weight = np.r_[frac, 1 - frac]

        self.clf = LogisticRegression(C=1e6)
        self.clf.fit(X2, labels, sample_weight=sample_weight)

    def predict(self, X):
        assert self.clf is not None
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        return self.clf.predict_proba(X)[:, 1] * self.scale

def exclude_coldstarts(weights: list[float], velocities: list[float]):
    """
    During warmups, athletes might just not follow their usual techniques
    when it comes to early warmups. Some athletes tend to just the bar off the flow
    Some just focus on getting a feel, and some would just actually use them to
    warmup their body.
    These early warmups might change the outcome when predicting the 1rm due to the
    way they are. So exclude them in our warmup series
    """
    max_weight = max(weights)
    not_cold = [(w, v) for w, v in zip(weights, velocities) if w / max_weight > 0.25]

    if not not_cold:
        return [], []

    filtered_weights, filtered_velocities = zip(*not_cold)
    return list(filtered_weights), list(filtered_velocities)

def mvt(exercise):
    mvt_map = {
        "bench press": 0.17,
        "squat": 0.30,
        "deadlift": 0.15,
        "overhead press": 0.20,
        "pull-up": 0.23,
        "prone row": 0.50,
    }
    return mvt_map.get(exercise.lower())

def predict_1rm_with_mvt(exercise, coeffs):
    MVT = mvt(exercise)

    shifted = np.array(coeffs, dtype=float)
    shifted[-1] -= MVT
    roots = np.roots(shifted)
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]

    if len(roots) == 0:
        return None

    return roots.max()