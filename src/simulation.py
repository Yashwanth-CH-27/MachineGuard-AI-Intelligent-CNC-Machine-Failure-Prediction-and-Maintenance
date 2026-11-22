"""Generate transparent, self-contained *simulated* machining observations.

These observations are NOT original AI4I records or real CNC sensor data.
The physical variables and example failure modes are loosely inspired by the
public AI4I 2020 description; values/labels are created independently here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_simulated_data(rows: int = 10000, seed: int = 2026) -> pd.DataFrame:
    if rows < 20:
        raise ValueError("At least 20 simulated records are required")
    rng = np.random.default_rng(seed)
    product = rng.choice(["L", "M", "H"], size=rows, p=[0.55, 0.30, 0.15])
    ambient = np.clip(300 + np.cumsum(rng.normal(0, .055, rows)), 294, 306)
    ambient = 300 + (ambient - np.mean(ambient)) * (2.0 / max(np.std(ambient), .01))
    process = ambient + 10 + rng.normal(0, 1.0, rows)
    # Operating states couple spindle speed and torque, as in machining.
    speed = np.clip(rng.normal(1550, 270, rows), 950, 3000).astype(int)
    torque = np.clip(41 - .031 * (speed - 1500) + rng.normal(0, 5.4, rows), 3.5, 85)
    # Wear counts reset periodically, not tied to true elapsed wall time.
    wear = np.empty(rows, dtype=int)
    elapsed = 0
    for i, t in enumerate(product):
        if elapsed > rng.integers(210, 245):
            elapsed = 0
        wear[i] = elapsed
        elapsed += {"L": 2, "M": 3, "H": 5}[t]
    heat_condition = ((process - ambient) < 8.3) & (speed < 1420)
    power_kw = torque * speed * (2 * np.pi / 60) / 1000
    power_condition = (power_kw < 3.25) | (power_kw > 9.25)
    wear_torque = wear * torque
    strain_limit = np.select([product == "L", product == "M"], [10500, 12000], default=13000)
    strain_condition = wear_torque > strain_limit
    # Probabilistic tool failure, adding noise instead of a perfect lookup rule.
    tool_condition = (wear > 205) & (rng.random(rows) < np.clip((wear - 200)/75, .02, .55))
    random_condition = rng.random(rows) < 0.0015
    conditions = heat_condition | power_condition | strain_condition | tool_condition | random_condition
    # Small label uncertainty makes this an imperfect supervised learning task.
    flip = rng.random(rows) < .003
    target = np.logical_xor(conditions, flip).astype(int)
    df = pd.DataFrame({
        "Type": product,
        "Air temperature [K]": np.round(ambient, 1),
        "Process temperature [K]": np.round(process, 1),
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": np.round(torque, 1),
        "Tool wear [min]": wear,
        "Machine failure": target,
    })
    return df


if __name__ == "__main__":
    from .constants import SIMULATED_DATA
    SIMULATED_DATA.parent.mkdir(exist_ok=True, parents=True)
    frame = make_simulated_data()
    frame.to_csv(SIMULATED_DATA, index=False)
    print(f"Saved {len(frame):,} simulated rows to {SIMULATED_DATA} ({frame['Machine failure'].sum()} positives)")
