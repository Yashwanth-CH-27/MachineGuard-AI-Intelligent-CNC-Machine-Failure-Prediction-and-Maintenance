import pandas as pd
import pytest
from src.constants import FEATURES


@pytest.fixture
def good_row():
    return pd.DataFrame([{
        "Type": "L", "Air temperature [K]": 300.2,
        "Process temperature [K]": 309.8,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }])[FEATURES]
