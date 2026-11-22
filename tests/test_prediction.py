import pandas as pd
import pytest
from src.modeling import load_artifact, load_metrics, scored_metrics
from src.predict import predict_many, priority_from_score, local_sensitivity, html_prediction_report
from src.history import save_predictions, load_history


def test_saved_model_produces_valid_probability(good_row):
    result = predict_many(good_row)
    assert result.shape[0] == 1
    assert 0 <= result.iloc[0]["Failure risk score"] <= 1
    assert result.iloc[0]["Review priority"] in {"Low", "Monitor", "High", "Critical"}
    assert result.iloc[0]["Predicted failure"] in {0,1}


def test_batch_prediction(good_row):
    batch = pd.concat([good_row, good_row], ignore_index=True)
    result = predict_many(batch)
    assert len(result) == 2
    assert result["Failure risk score"].notna().all()


def test_extra_column_excluded_from_model(good_row):
    good_row["TWF"] = 1  # deliberate trap: must never be used as prediction feature
    baseline = good_row.drop(columns=["TWF"])
    assert predict_many(good_row)["Failure risk score"].equals(predict_many(baseline)["Failure risk score"])


def test_local_sensitivity_complete(good_row):
    sensitivity = local_sensitivity(good_row)
    assert len(sensitivity) == 6
    assert all("score_change_if_typical" in row for row in sensitivity)


def test_html_report_encodes_untrusted_text(good_row):
    outcome = predict_many(good_row).iloc[0]
    altered = good_row.iloc[0].copy()
    altered["Type"] = "<script>alert(1)</script>"
    html = html_prediction_report(altered, outcome, load_artifact())
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_priorities_deterministic():
    assert priority_from_score(.01, .5) == "Low"
    assert priority_from_score(.75, .5) == "Critical"
    assert priority_from_score(.51, .5) == "High"
    assert priority_from_score(.35, .5) == "Monitor"


def test_saved_metrics_are_real_and_consistent():
    report = load_metrics()
    assert report["split"]["train"] + report["split"]["validation"] + report["split"]["test"] == report["data"]["rows"]
    assert report["test_metrics"]["confusion_matrix"]
    assert report["selected_model"] in {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert 0 < report["test_metrics"]["average_precision"] <= 1


def test_history_file_round_trip(tmp_path, good_row):
    results = predict_many(good_row)
    path = tmp_path / "history.csv"
    save_predictions(results, path=path, mode="Test")
    found = load_history(path)
    assert found.shape[0] == 1
    assert found.iloc[0]["Mode"] == "Test"
    assert found.iloc[0]["Failure risk score"] == pytest.approx(results.iloc[0]["Failure risk score"])


def test_saved_probability_not_hardcoded(good_row):
    a = good_row.copy()
    b = good_row.copy()
    b.loc[0, "Rotational speed [rpm]"] = 1150
    b.loc[0, "Process temperature [K]"] = 306.0
    a_score = float(predict_many(a)["Failure risk score"].iloc[0])
    b_score = float(predict_many(b)["Failure risk score"].iloc[0])
    assert a_score != b_score


def test_reloaded_saved_artifact_matches_active_dataset():
    from src.data import file_sha256, dataset_path
    import sklearn
    a = load_artifact()
    assert a["dataset_hash"] == file_sha256(dataset_path())
    assert a["sklearn_version"] == sklearn.__version__


def test_batch_rejects_more_than_5000(good_row):
    from src.data import InputValidationError
    with pytest.raises(InputValidationError, match="at most"):
        predict_many(pd.concat([good_row] * 5001, ignore_index=True))
