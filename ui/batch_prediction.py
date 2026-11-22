"""Validated CSV batch predictions and export."""
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from src.constants import FEATURES, ROOT
from src.data import validate_measurements, InputValidationError
from src.history import save_predictions
from src.predict import predict_many
from .common import heading, provenance_notice


def render(artifact, report):
    heading("BATCH ANALYSIS", "Analyze multiple machine records", "Upload a CSV with the six required columns; download predictions for every valid row.")
    provenance_notice(artifact)
    sample_path = ROOT / "data" / "sample_input.csv"
    st.download_button("Download sample CSV template", sample_path.read_bytes(), file_name="machineguard_sample_input.csv", mime="text/csv")
    st.code(", ".join(FEATURES), language="text")
    uploaded = st.file_uploader("Upload a machine measurement CSV", type=["csv"], help="Maximum 5,000 rows. Extra columns will be preserved in the result but never used as model inputs.")
    if uploaded is None:
        st.info("Use the sample template to verify the required column names, then upload your file.")
        return
    if uploaded.size > 2_000_000:
        st.error("File exceeds the 2 MB upload limit.")
        return
    try:
        raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        validated = validate_measurements(raw)
    except (InputValidationError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        st.error(f"CSV validation failed: {exc}")
        return
    st.write(f"**{len(validated):,} rows validated** · Predictions are local and are not sent to an online API.")
    if st.button("Run batch inspection", type="primary"):
        try:
            results = predict_many(raw, artifact)
            save_predictions(results, mode="CSV batch")
            st.session_state["batch_result"] = (uploaded.name, results)
        except (ValueError, InputValidationError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("batch_result")
    if not saved or saved[0] != uploaded.name:
        return
    results = saved[1]
    a, b, c = st.columns(3)
    a.metric("Records analyzed", len(results))
    b.metric("Elevated-risk classifications", int(results["Predicted failure"].sum()))
    c.metric("Average risk score", f"{results['Failure risk score'].mean():.1%}")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.plotly_chart(px.histogram(results, x="Failure risk score", nbins=25, title="Distribution of model risk scores"), use_container_width=True)
    st.download_button("Download CSV prediction results", results.to_csv(index=False), file_name="machineguard_batch_predictions.csv", mime="text/csv", use_container_width=True)
