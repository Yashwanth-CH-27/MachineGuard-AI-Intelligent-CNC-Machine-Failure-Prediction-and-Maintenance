"""MachineGuard AI — educational CNC failure-risk inspection application."""
from pathlib import Path
import streamlit as st
from src.constants import ROOT
from src.modeling import load_artifact, load_metrics
from src.data import source_label
from ui import dashboard, single_prediction, batch_prediction, model_insights, about

st.set_page_config(page_title="MachineGuard AI | CNC Maintenance", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained maintenance model...")
def cached_model():
    return load_artifact()

@st.cache_data(show_spinner=False)
def cached_metrics():
    return load_metrics()

try:
    artifact = cached_model()
    results = cached_metrics()
except Exception as exc:
    st.error(f"Cannot load trained model: {exc}")
    st.code("python bootstrap.py --force-train")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand-icon">⚙️</div>', unsafe_allow_html=True)
    st.title("MachineGuard AI")
    st.caption("CNC FAILURE RISK & MAINTENANCE")
    st.divider()
    selected = st.radio("Navigation", [
        "Overview", "Check a Machine", "Batch Inspection", "Model Intelligence", "Project Guide"
    ], label_visibility="collapsed")
    st.divider()
    st.caption("ACTIVE MODEL")
    st.write(f"**{artifact['model_name']}**")
    st.caption(source_label(artifact["dataset_kind"]))
    st.caption("Local educational demonstration · No live sensor feed")

handlers = {
    "Overview": dashboard.render,
    "Check a Machine": single_prediction.render,
    "Batch Inspection": batch_prediction.render,
    "Model Intelligence": model_insights.render,
    "Project Guide": about.render,
}
handlers[selected](artifact, results)

st.divider()
st.caption("MachineGuard AI · Student academic project · Synthetic records are not real operational measurements. Never use these scores as safety decisions.")
