import numpy as np
import pandas as pd
import pytest
from src.constants import FEATURES, TARGET
from src.data import validate_measurements, InputValidationError, load_training_data
from src.simulation import make_simulated_data
from src.preprocessing import MachiningFeatures


def test_valid_input(good_row):
    clean = validate_measurements(good_row)
    assert list(clean.columns) == FEATURES
    assert clean.shape == (1, 6)


def test_invalid_type(good_row):
    good_row.loc[0, "Type"] = "W"
    with pytest.raises(InputValidationError, match="product type"):
        validate_measurements(good_row)


def test_missing_column(good_row):
    with pytest.raises(InputValidationError, match="Missing"):
        validate_measurements(good_row.drop(columns=["Torque [Nm]"]))


def test_text_in_number(good_row):
    good_row["Torque [Nm]"] = good_row["Torque [Nm]"].astype(object)
    good_row.loc[0, "Torque [Nm]"] = "xxx"
    with pytest.raises(InputValidationError, match="must contain numbers"):
        validate_measurements(good_row)


def test_nan_or_infinite_blocked(good_row):
    good_row.loc[0, "Air temperature [K]"] = np.inf
    with pytest.raises(InputValidationError, match="Invalid"):
        validate_measurements(good_row)


def test_out_of_range(good_row):
    good_row.loc[0, "Tool wear [min]"] = -10
    with pytest.raises(InputValidationError, match="Invalid"):
        validate_measurements(good_row)


def test_simulation_reproducible():
    a = make_simulated_data(500, seed=12)
    b = make_simulated_data(500, seed=12)
    pd.testing.assert_frame_equal(a, b)
    assert a[TARGET].nunique() == 2
    assert set(a["Type"]) <= {"L", "M", "H"}


def test_engineered_values(good_row):
    added = MachiningFeatures().fit_transform(good_row)
    assert added.iloc[0]["Temperature difference [K]"] == pytest.approx(9.6)
    assert added.iloc[0]["Spindle power [kW]"] == pytest.approx(6.283185, abs=.00001)
    assert added.iloc[0]["Wear torque [min Nm]"] == 4000


def test_training_data_no_leakage_columns():
    frame, summary = load_training_data()
    assert list(frame.drop(columns=[TARGET]).columns) == FEATURES
    assert "TWF" not in frame.columns
    assert "Product ID" not in frame.columns
    assert summary["rows"] == 10000
