import joblib
import pandas as pd
from pathlib import Path

MEAN_PEAK_MODEL_PATH = Path("./assets/models/mean_peak_rpe_model.joblib")
mean_peak_model = joblib.load(MEAN_PEAK_MODEL_PATH)

RPE = ["<5", "6", "7", "8", "9", "10"]

def estimate_rpe_with_velocity(mean_velocity, peak_velocity):
    X = pd.DataFrame({"mean_velocity_ms": [mean_velocity], "peak_velocity_ms": [peak_velocity]})
    y = mean_peak_model.predict(X)
    return RPE[y[0]]
