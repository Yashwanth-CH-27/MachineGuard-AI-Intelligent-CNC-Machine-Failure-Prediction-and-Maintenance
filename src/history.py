"""Simple local file-based log; deliberately excluded from GitHub."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .constants import HISTORY_PATH


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def save_predictions(frame: pd.DataFrame, path: Path = HISTORY_PATH, mode: str = "Manual"):
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = frame.copy()
    new.insert(0, "Recorded UTC", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    new.insert(1, "Mode", mode)
    existing = load_history(path)
    combined = pd.concat([existing, new], ignore_index=True).tail(5000)
    # Replace only after contents are serializable; log never becomes training data.
    temporary = path.with_suffix(".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
