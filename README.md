# ⚙️ MachineGuard AI

### Intelligent CNC Machine Failure Prediction and Maintenance

**An interactive, reproducible student ML application** for estimating failure risk from industrial machining process measurements. Built with **Streamlit + scikit-learn** and packaged with a **Windows one-click launcher**.

> **Data truth:** This ZIP already contains a functional model trained on **independently simulated data**. On the first launch, it tries to download the real **UCI AI4I 2020 *synthetic* benchmark** and automatically retrains if downloading succeeds. If the official download fails or the PC is offline, the included demo runs without that download. These are not live CNC sensor readings, and risk estimates are **not** validated safety assessments.

## Quick start — Windows

1. Install **Python 3.10–3.13** from https://www.python.org/downloads/ (enable **Add Python to PATH**); an internet connection is needed to install dependencies the first time.
2. Extract the ZIP to a normal writable folder (for example `Documents\MachineGuard-AI`).
3. Double-click **`START_PROJECT.bat`**. It creates `.venv`, installs the required Python packages, checks the dataset and model, and starts Streamlit.
4. Your default browser opens to `http://localhost:8501`. Keep the command window open. Press **Ctrl+C** in that window to stop the local server.

First launch may take longer because packages install. After setup, package installation is skipped on later launches. If you have no internet on the first run, Python packages must already be installed in the virtual environment; the simulated dataset and pretrained model themselves do not require internet. No paid APIs, API keys, cloud database, Docker, or GPU are used.

## What problem does it solve?

CNC machining operations involve thermal conditions, spindle speed, torque, and tool wear. Combinations of these variables can be associated with machinery failures. MachineGuard AI supports an operator's inspection workflow by scoring each observation and presenting **model-derived failure risk**, a **heuristic inspection priority**, and transparent model insights.

### Features

- **Overview:** dataset information, real training metrics, failure-label charts, spindle/torque scatterplot, local inspection history.
- **Check a Machine:** interactive measurements, validation, result class, model score, heuristic review priority, local what-if sensitivity, and downloadable **HTML report**.
- **Batch Inspection:** sample CSV template, up to 5,000 rows with strict schema and ranges, score predictions, histogram, downloadable results **CSV**.
- **Model Intelligence:** comparison of three fitted algorithms on validation data, separate held-out test metrics and confusion matrix, validation permutation importance.
- **Project Guide:** explanations, model workflow, provenance, and limitations.

## Technology and pipeline

| Layer | Technology |
|---|---|
| Language | Python 3.10–3.13 |
| Data manipulation | Pandas, NumPy |
| Model training | scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Score calibration | 3-fold sigmoid calibration on training partition |
| Modeling | train/validation/test stratified split 60%/20%/20% |
| UI & charts | Streamlit, Plotly, custom CSS |
| Model storage | Joblib |
| History | Local CSV, excluded from Git |
| Tests | Pytest |

The model uses **exactly six raw operating inputs** (1 categorical + 5 numeric). Feature engineering adds temperature difference, approximate spindle power, and wear × torque. Identifiers and failure-mode outcome columns are excluded to prevent leakage. Each preprocessing transform is fitted within cross-validated training pipelines. Model selection is by **validation average precision** and its score cutoff by **validation F1**; the test set is held out until final evaluation. Scores are calibrated estimates from training data, **not trustworthy real-world calibrated probabilities**.

## Actual example input

```csv
Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
L,302.0,309.0,1300,42.0,170
```

The result is calculated by the trained model and depends on the active dataset. No particular prediction is promised for an illustrative input.

## Dataset and actual model evaluation

- Included: `data/machining_simulated.csv` — **independently generated simulation**, 10,000 rows; reproducible with `python -m src.simulation`.
- Optional download: `data/ai4i2020_official.csv` — the UCI AI4I 2020 published **synthetic** benchmark, CC BY 4.0; automatic first-run attempt with `bootstrap.py`.
- Details / attribution: [data/DATASET_ATTRIBUTION.md](data/DATASET_ATTRIBUTION.md).
- The current saved model's training source is displayed prominently throughout the app and in `models/training_results.json`.
- For genuinely measured scores, open `models/training_results.json` or the Model Intelligence page. **Never reuse the simulator's scores to claim UCI or industrial performance.**

## Manual commands (alternative to BAT)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python bootstrap.py --prefer-official
python -m streamlit run app.py
```

To retry fetching the original UCI dataset: `python bootstrap.py --refresh-official`. To retrain: `python bootstrap.py --force-train`. Run checks with `python -m pip install -r requirements-dev.txt` and `python -m pytest -q`.

## Folder structure

```text
MachineGuard-AI/
├── START_PROJECT.bat          # Windows launcher: click once
├── launcher.py                # starts Streamlit / opens browser
├── bootstrap.py               # optional UCI fetch; safe fallback; train if needed
├── app.py                     # app shell and navigation
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/config.toml
├── assets/styles.css
├── data/
│   ├── machining_simulated.csv   # disclosed independent demo dataset
│   ├── sample_input.csv
│   └── DATASET_ATTRIBUTION.md
├── models/
│   ├── trained_model.joblib      # pre-trained simulator model
│   └── training_results.json    # authentic metrics for that model
├── src/
│   ├── constants.py, simulation.py, data.py
│   ├── preprocessing.py, modeling.py, predict.py, history.py
├── ui/
│   ├── dashboard.py, single_prediction.py, batch_prediction.py
│   ├── model_insights.py, about.py, common.py
├── notebooks/exploratory_analysis.ipynb
├── tests/                   # runnable pytest checks
├── docs/                    # report, design, viva, GitHub, test log
└── runtime/                 # locally generated inspection history; excluded from Git
```

## Inspect in a viva

Open **Overview** → enter an example in **Check a Machine** → save HTML report → download sample CSV and upload it in **Batch Inspection** → open **Model Intelligence** → discuss class imbalance, data leakage, PR-AUC/recall, and simulator limitations.

## Limitations and safe use

- Only a **machining-process fault/risk classification** demonstration. Not validated for particular CNC makes or models, and no specific fault-mode or remaining-useful-life prediction.
- Not connected to physical machines. No continuous sensor polling. `Product Type` is product category, **not a CNC machine ID**.
- Maintenance review labels are **heuristics** on top of the fitted risk score; they are not validated service schedules.
- The official dataset and the offline fallback are **synthetic**. Random row splits and synthetic failure rules do not prove transfer to factory operations. Human inspections and real telemetry would be required for operational use.
- Fonts are optional web-loaded; the dashboard works with system fonts if offline.

## Troubleshooting

- **`Python not found`**: install Python 3.10–3.13 with PATH enabled, restart Command Prompt, and run the BAT again.
- **`pip` or network error**: use a stable internet connection on the first launch; dependencies are not embedded in the ZIP. Remove `.venv/.deps_installed` to force dependency installation if needed.
- **Port 8501 in use**: close other Streamlit sessions or edit `launcher.py` port and address together.
- **UCI download unavailable**: this is nonfatal. The included locally simulated data and fitted model work. Retry with `.venv\Scripts\python.exe bootstrap.py --refresh-official` if desired.
- **Model load error**: `.venv\Scripts\python.exe bootstrap.py --force-train` from the extracted folder.
- **Stale batch result**: upload the updated file with a different name or rerun after changing its filename.

## Originality / license / GitHub

All application implementation here is project-specific and no existing project's repository is copied. The independent simulation is fully defined in code; public AI4I is explicitly cited. Original source code: **MIT** (see LICENSE); optional UCI AI4I CSV: **CC BY 4.0** (see dataset attribution). No secrets, generated history, or virtual environment are intended to be uploaded. [GitHub submission walkthrough](docs/GITHUB_SUBMISSION.md).

## More documentation

- [Academic project report](docs/PROJECT_REPORT.md)
- [Technical architecture](docs/TECHNICAL_DOCUMENTATION.md)
- [Viva questions and answers](docs/VIVA_QUESTIONS.md)
- [GitHub instructions](docs/GITHUB_SUBMISSION.md)
- [Testing and verification](docs/TEST_RESULTS.md)
- [Quick demonstration script](docs/QUICK_START.md)

**No GitHub URL is invented.** Create your repository and paste the real URL into the college Excel sheet.
