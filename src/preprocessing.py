"""Leakage-safe feature engineering and preprocessing fit inside sklearn pipelines."""
from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from .constants import FEATURES, NUMERIC_FEATURES

ENGINEERED = ["Temperature difference [K]", "Spindle power [kW]", "Wear torque [min Nm]"]


class MachiningFeatures(BaseEstimator, TransformerMixin):
    """Physical derived quantities built only from the six permitted inputs."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=FEATURES).copy()
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col])
        df[ENGINEERED[0]] = df["Process temperature [K]"] - df["Air temperature [K]"]
        df[ENGINEERED[1]] = (df["Rotational speed [rpm]"] * df["Torque [Nm]"] * 2 * np.pi / 60) / 1000
        df[ENGINEERED[2]] = df["Tool wear [min]"] * df["Torque [Nm]"]
        return df


def make_preprocessor():
    numerical = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numerical, NUMERIC_FEATURES + ENGINEERED),
        ("cat", categorical, ["Type"]),
    ])
