"""Landing dashboard with real dataset summaries, model scores and saved user history."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data import load_training_data
from src.history import load_history
from .common import heading, provenance_notice


def render(artifact, report):
    heading("COMMAND CENTER / OVERVIEW", "Industrial maintenance at a glance", "Review model validation, machining data, and your inspection history in one place.")
    provenance_notice(artifact)
    source = report["data"]
    scores = report["test_metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Dataset records", f"{source['rows']:,}")
    b.metric("Labeled failures", f"{source['failure_rows']:,}", f"{source['failure_rate']:.1%} prevalence", delta_color="off")
    c.metric("Held-out F1", f"{scores['f1']:.3f}")
    d.metric("Held-out recall", f"{scores['recall']:.1%}")
    st.markdown("### Machine operation patterns")
    data, _ = load_training_data()
    data["Failure state"] = data["Machine failure"].map({0: "Normal label", 1: "Failure label"})
    left, right = st.columns(2)
    with left:
        frame = pd.DataFrame({"Outcome": ["Normal", "Failure"], "Count": [int((data['Machine failure']==0).sum()), int((data['Machine failure']==1).sum())]})
        fig = px.bar(frame, x="Outcome", y="Count", color="Outcome", title="Dataset label distribution", color_discrete_map={"Normal": "#4b8cca", "Failure": "#ee8b5b"})
        fig.update_layout(showlegend=False, height=350, margin=dict(t=54, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            data.sample(min(2500, len(data)), random_state=42),
            x="Rotational speed [rpm]", y="Torque [Nm]", color="Failure state",
            title="Spindle speed vs torque (sample of records)", opacity=.67,
            color_discrete_map={"Normal label": "#4b8cca", "Failure label": "#ee8b5b"},
        )
        fig.update_layout(height=350, margin=dict(t=54, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Your inspection log")
    history = load_history()
    if history.empty:
        st.info("No inspections saved yet. Open **Check a Machine** to make your first prediction.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Saved inspections", f"{len(history):,}")
        col2.metric("Elevated-risk classifications", str(int(history["Predicted failure"].sum())))
        col3.metric("Critical review priority", str(int((history["Review priority"] == "Critical").sum())))
        st.dataframe(history.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Download inspection history", history.to_csv(index=False), file_name="machineguard_inspection_history.csv", mime="text/csv")
