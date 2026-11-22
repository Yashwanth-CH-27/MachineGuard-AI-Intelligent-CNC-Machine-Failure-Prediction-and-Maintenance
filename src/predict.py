"""Batch and single equipment prediction using the *saved real fitted model*."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
import pandas as pd
from .constants import FEATURES
from .data import validate_measurements, source_label
from .modeling import load_artifact


def priority_from_score(score, threshold):
    """Transparent operator-review heuristic, not a learned maintenance policy."""
    if score >= max(.7, threshold + .25):
        return "Critical"
    if score >= threshold:
        return "High"
    if score >= max(.1, threshold * .5):
        return "Monitor"
    return "Low"


def predict_many(frame, artifact=None):
    artifact = artifact or load_artifact()
    checked = validate_measurements(frame)
    probabilities = artifact["model"].predict_proba(checked)[:, 1]
    threshold = float(artifact["threshold"])
    outcome = frame.reset_index(drop=True).copy()
    outcome["Failure risk score"] = probabilities.round(5)
    outcome["Predicted failure"] = (probabilities >= threshold).astype(int)
    outcome["Review priority"] = [priority_from_score(float(p), threshold) for p in probabilities]
    return outcome


def local_sensitivity(checked_row, artifact=None):
    """Change one raw feature to the training median/mode and measure score delta.

    This is a what-if sensitivity test and NOT a causal diagnosis or SHAP value.
    """
    artifact = artifact or load_artifact()
    frame = validate_measurements(checked_row)
    if len(frame) != 1:
        raise ValueError("Local explanation only supports one machine record")
    original = float(artifact["model"].predict_proba(frame)[0, 1])
    details = []
    for col in FEATURES:
        trial = frame.copy()
        reference = artifact["typical_values"][col]
        trial.loc[trial.index[0], col] = reference
        hypothetical = float(artifact["model"].predict_proba(trial)[0, 1])
        details.append({
            "feature": col,
            "score_change_if_typical": round(hypothetical - original, 4),
            "typical_value": reference,
        })
    return sorted(details, key=lambda entry: abs(entry["score_change_if_typical"]), reverse=True)


def html_prediction_report(row, outcome, artifact):
    values = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(row[k]))}</td></tr>" for k in FEATURES)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MachineGuard AI - Prediction Report</title>
<style>body{{font:16px system-ui;max-width:850px;margin:40px auto;padding:24px;color:#243243}} h1{{color:#174065}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #dde3ea;text-align:left}}.tag{{background:#eef4fa;padding:13px;border-radius:8px}}small{{color:#586777}}</style></head><body>
<h1>MachineGuard AI · Inspection Report</h1><p><small>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</small></p>
<p class="tag"><strong>Estimated failure risk:</strong> {float(outcome['Failure risk score'])*100:.1f}% &nbsp; | &nbsp;
<strong>Review priority:</strong> {escape(str(outcome['Review priority']))} &nbsp; | &nbsp; <strong>Predicted class:</strong> {int(outcome['Predicted failure'])}</p>
<h2>Input measurements</h2><table>{values}</table>
<h2>Model information</h2><p>Algorithm: {escape(artifact['model_name'])} (calibrated). Classification threshold: {artifact['threshold']:.2f}. Data source: {escape(source_label(artifact['dataset_kind']))}.</p>
<p><small>Educational decision support only. Not connected to live machinery. This score is not a diagnosis, time-to-failure estimate, or authorization to stop a machine. Maintenance priorities are heuristic; have a qualified operator inspect any safety concerns.</small></p>
</body></html>"""
