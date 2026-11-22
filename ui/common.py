"""Shared interface elements."""
import streamlit as st
from src.data import source_label


def heading(kicker, headline, subtitle):
    st.markdown(f'<p class="eyebrow">{kicker}</p>', unsafe_allow_html=True)
    st.title(headline)
    st.write(subtitle)


def provenance_notice(artifact):
    text = source_label(artifact["dataset_kind"])
    if artifact["dataset_kind"] != "official_uci_synthetic":
        st.warning(f"**Demo data:** {text}. Scores are measured on simulated examples generated with this project; they are not results on the original UCI benchmark or real machines.")
    else:
        st.info(f"**Dataset:** {text}. This published dataset is itself synthetic, not a live factory dataset.")
