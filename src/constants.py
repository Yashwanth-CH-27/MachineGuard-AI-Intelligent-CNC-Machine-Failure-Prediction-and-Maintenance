"""Single source of truth for the machining measurements used by the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
METRICS_PATH = ROOT / "models" / "training_results.json"
HISTORY_PATH = ROOT / "runtime" / "prediction_history.csv"
OFFICIAL_DATA = DATA_DIR / "ai4i2020_official.csv"
SIMULATED_DATA = DATA_DIR / "machining_simulated.csv"

TARGET = "Machine failure"
FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURES[1:]
TYPE_OPTIONS = ("L", "M", "H")
# Deliberately permissive physical plausibility checks, not learned limits.
LIMITS = {
    "Air temperature [K]": (250, 380),
    "Process temperature [K]": (250, 450),
    "Rotational speed [rpm]": (100, 10000),
    "Torque [Nm]": (0, 300),
    "Tool wear [min]": (0, 1000),
}
EXCLUDED = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
OFFICIAL_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip",
]
