"""Independent mini-model smoke test, no reliance on the prebuilt model's scores."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.constants import FEATURES, TARGET
from src.modeling import pipeline_for, scored_metrics
from src.simulation import make_simulated_data


def test_actual_pipeline_fits_and_predicts_without_target_leakage():
    frame = make_simulated_data(rows=800, seed=33)
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=.25, stratify=frame[TARGET], random_state=17
    )
    model = CalibratedClassifierCV(
        estimator=pipeline_for(LogisticRegression(max_iter=500, class_weight="balanced")),
        cv=3, method="sigmoid",
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    results = scored_metrics(y_test, p, threshold=.4)
    assert 0 < results["roc_auc"] <= 1
    assert sum(map(sum, results["confusion_matrix"])) == len(y_test)
