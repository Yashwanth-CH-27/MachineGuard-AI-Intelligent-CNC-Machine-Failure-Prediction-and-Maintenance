"""Human-readable project explanation and deployment limitations."""
import streamlit as st
from .common import heading, provenance_notice


def render(artifact, report):
    heading("PROJECT GUIDE", "What MachineGuard AI actually does", "A student-built inspection and educational analytics platform for simulated industrial machining conditions.")
    provenance_notice(artifact)
    st.markdown("""
### The problem
Industrial CNC-style machining involves spindle rotation, cutting forces, tool wear, and thermal conditions. Abnormal combinations can coincide with failures. Operators benefit from understandable risk indicators and a way to inspect multiple measurement records.

### The solution
MachineGuard AI accepts six inputs—product variant (L/M/H), air temperature, process temperature, spindle RPM, torque, and accumulated tool wear. It applies a trained binary classifier to estimate how closely those conditions resemble recorded *failure labels*. The dashboard offers manual inspection, CSV batch classification, exportable reports, and model evaluations.

### Real example
Imagine a CNC operator measuring 302 K air temperature, 309 K process temperature, 1,300 RPM, 42 Nm torque and 170 minutes of wear. They enter those values, review the model's estimate, and decide whether a trained technician should investigate. The estimate is **not** a confirmed fault diagnosis or a statement about the remaining lifetime of the equipment.

### Data and evaluation
The [AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) contains 10,000 simulated industrial machining records (DOI 10.24432/C5HS5C; CC BY 4.0). When accessible on first run, MachineGuard can download those original synthetic records. If offline, it uses the **separate, locally simulated** demonstration data shipped in this ZIP. Both are synthetic; neither is a live CNC machine feed.

### ML workflow
1. Check required columns, missing values and duplicates.
2. Split labels into train / validation / test (60% / 20% / 20%).
3. Engineer temperature difference, approximate spindle power, wear × torque using allowed inputs.
4. Train and calibrate Logistic Regression, Decision Tree, and Random Forest with 3-fold calibration.
5. Select by validation average precision and choose classification threshold by validation F1.
6. Evaluate once on held-out test data and save the model locally.
7. Use saved pipeline for predictions; report input sensitivity and validation permutation importance.

### Important limitations
- This application does **not** diagnose the specific mechanical fault, connect to hardware, or continuously monitor machines.
- Its product type field is a product quality class, **not** a make/model of CNC machine.
- It has no time-to-failure target. "Predictive maintenance" here means risk-oriented decision support, not predicting the exact future breakdown date.
- Risk bands and review priorities are transparent heuristics, **not** validated safety recommendations.
- Synthetic results may appear impressive while failing to transfer to real factory conditions.
- To use it in industry, gather representative real sensor data, validate calibration and safety, and obtain expert authorization.

### Technologies
Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly · Joblib · Pytest.
""")
    st.write("**Selected fitted algorithm:**", artifact["model_name"])
    st.write("**Data source for the saved model:**", report["data"]["dataset_kind"])
