"""Operator-friendly single-machine measurement form and downloadable HTML report."""
from datetime import datetime
import pandas as pd
import streamlit as st
from src.data import validate_measurements, InputValidationError
from src.history import save_predictions
from src.predict import html_prediction_report, predict_many, local_sensitivity
from .common import heading, provenance_notice


def render(artifact, report):
    heading("SINGLE EQUIPMENT CHECK", "Inspect a CNC machining operation", "Provide current conditions to estimate the class and risk score learned by the model.")
    provenance_notice(artifact)
    with st.form("inspection_form"):
        left, right = st.columns(2)
        with left:
            machine = st.text_input("Machine reference (optional)", value="CNC-001", help="Local label used only in the downloaded report filename")
            kind = st.selectbox("Product variant", ["L", "M", "H"], index=0, help="L / M / H are product quality variants in the dataset, not different machine models")
            air = st.number_input("Air temperature (Kelvin)", min_value=250.0, max_value=380.0, value=300.0, step=.1)
            process = st.number_input("Process temperature (Kelvin)", min_value=250.0, max_value=450.0, value=310.0, step=.1)
        with right:
            rpm = st.number_input("Spindle speed (RPM)", min_value=100, max_value=10000, value=1530, step=10)
            torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=300.0, value=40.0, step=.5)
            wear = st.number_input("Accumulated tool wear (minutes)", min_value=0, max_value=1000, value=110, step=5)
            st.caption("The values are checked against broad physical ranges; they are not measured by this app.")
        submitted = st.form_submit_button("Run equipment inspection", type="primary", use_container_width=True)
    if submitted:
        input_row = pd.DataFrame([{
            "Type": kind, "Air temperature [K]": air, "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear,
        }])
        try:
            checked = validate_measurements(input_row)
            outcome = predict_many(checked, artifact)
            save_predictions(outcome, mode="Manual")
            st.session_state["single_inspection"] = (checked, outcome, machine)
        except (InputValidationError, ValueError) as exc:
            st.error(str(exc))
    if "single_inspection" not in st.session_state:
        st.info("Enter operating values and press **Run equipment inspection** to view a predicted outcome.")
        return
    checked, outcomes, machine = st.session_state["single_inspection"]
    answer = outcomes.iloc[0]
    risk = float(answer["Failure risk score"])
    st.markdown("### Inspection result")
    first, second, third = st.columns(3)
    first.metric("Model risk score", f"{risk:.1%}")
    second.metric("Predicted class", "Elevated risk" if answer["Predicted failure"] else "No failure flagged")
    third.metric("Review priority", answer["Review priority"])
    st.progress(max(0., min(1., risk)))
    threshold = float(artifact["threshold"])
    st.caption(f"The model classifies a failure when score ≥ {threshold:.1%} (threshold tuned on validation data). Scores are calibrated estimates, not guaranteed real-world probabilities.")
    if answer["Predicted failure"]:
        st.warning("Consider reviewing this equipment's measurements and scheduling a qualified maintenance inspection. This is not an automatic shutdown recommendation.")
    else:
        st.success("This record is not flagged by the fitted model. Continue ordinary monitoring; no model can guarantee that machinery is safe.")
    st.markdown("#### What-if factors")
    st.write("For each factor, the model substitutes the training-set typical value and calculates how much the score changes. This is **sensitivity analysis, not a causal failure diagnosis**.")
    influence = local_sensitivity(checked, artifact)
    st.dataframe(pd.DataFrame(influence).rename(columns={"feature": "Measurement", "typical_value": "Typical training value", "score_change_if_typical": "Score change if typical"}), hide_index=True, use_container_width=True)
    safe_machine = "".join(ch for ch in machine if ch.isalnum() or ch in "-_ ")[:40].strip().replace(" ", "_") or "CNC"
    st.download_button("Download HTML inspection report", data=html_prediction_report(checked.iloc[0], answer, artifact), file_name=f"MachineGuard_{safe_machine}_{datetime.now().strftime('%Y%m%d_%H%M')}.html", mime="text/html", use_container_width=True)
