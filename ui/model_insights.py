"""Honest model comparison and untampered final held-out evaluation."""
import pandas as pd
import plotly.express as px
import streamlit as st
from .common import heading, provenance_notice


def render(artifact, report):
    heading("MODEL INTELLIGENCE", "Transparent ML performance", "Scores below were created during an actual local training run; the final test partition was not used to select the model or threshold.")
    provenance_notice(artifact)
    a, b, c, d = st.columns(4)
    final = report["test_metrics"]
    a.metric("Test precision", f"{final['precision']:.1%}")
    b.metric("Test recall", f"{final['recall']:.1%}")
    c.metric("Test F1", f"{final['f1']:.3f}")
    d.metric("Test ROC-AUC", f"{final['roc_auc']:.3f}")
    st.markdown("### Algorithm comparison (validation set)")
    comparisons = pd.DataFrame(report["validation_comparison"])
    st.dataframe(comparisons, use_container_width=True, hide_index=True)
    fig = px.bar(comparisons.melt(id_vars="model", value_vars=["average_precision", "roc_auc", "f1"], var_name="Metric", value_name="Score"), x="model", y="Score", color="Metric", barmode="group", title="Model selection comparison (validation, not test)")
    fig.update_layout(height=410, yaxis_range=[0, 1.03])
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Selected **{report['selected_model']}** using {report['selection_basis']}. Threshold of {report['decision_threshold']:.2f} chosen using validation F1.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Held-out confusion matrix")
        cm = report["test_metrics"]["confusion_matrix"]
        matrix = pd.DataFrame(cm, index=["Actual normal", "Actual failure"], columns=["Predicted normal", "Predicted failure"])
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows are true labels; columns are model classifications. False negatives are failures missed by the model.")
    with right:
        st.markdown("### Permutation importance")
        imp = pd.DataFrame(report["validation_permutation_importance"])
        fig2 = px.bar(imp.sort_values("mean_drop_in_average_precision"), y="feature", x="mean_drop_in_average_precision", orientation="h", title="Loss of validation average precision when shuffled")
        fig2.update_layout(height=360, xaxis_title="Average precision decrease", yaxis_title="Input measurement")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Model dependence, not a causal explanation. Negative drops can arise from noisy evaluation.")
    st.markdown("### Training and evaluation audit")
    st.write(f"Train: **{report['split']['train']:,}** rows · Validation: **{report['split']['validation']:,}** rows · Test: **{report['split']['test']:,}** rows")
    st.write(f"Test accuracy: **{final['accuracy']:.4f}** · Average precision: **{final['average_precision']:.4f}** · Brier loss: **{final['brier_score']:.4f}**")
    st.caption("Scores come from a stratified random row split of synthetic observations. Rows may share simulation conditions; deployment generalization is not established. High accuracy alone is misleading when failures are rare.")
