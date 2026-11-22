# Technical documentation — MachineGuard AI

## 1. Architecture

```text
Windows START_PROJECT.bat
    ├── Python venv / dependency install (first launch)
    ├── bootstrap.py
    │      ├── dataset source: official UCI CSV, if obtained; else included independent simulator
    │      ├── data validation and SHA-256 model/data consistency check
    │      └── modeling.train_and_save -> joblib + JSON metrics if needed
    └── launcher.py -> local Streamlit server -> default browser
                                       |
                                       +-- Overview / individual / batch / model insights / guide
                                       +-- src.predict -> sklearn fitted calibrated pipeline
                                       +-- src.history -> runtime/prediction_history.csv
```

No remote prediction endpoint, user login, database service, Docker image, GPU or paid API is involved.

## 2. Dataset inputs

Required raw features: `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`. The target: `Machine failure` (binary 0/1). 10,000 rows when using either included simulation or the downloaded original AI4I data. AI4I's `UDI`, `Product ID` and failure-mode targets `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are excluded from the input pipeline. A product quality variant is **not** the machine type.

Dataset switching precedence: `data/ai4i2020_official.csv` exists → prefer official public synthetic records; otherwise → `data/machining_simulated.csv`. SHA-256 of the active CSV is saved alongside the model and checked on each launch to trigger automatic retraining when data source changes.

The included generator produces correlated physical measurements and stochastic failure labels based on documented example mechanisms. This **does not recreate AI4I rows**. The source is explicitly disclosed in the UI, saved model and metrics JSON; do not confuse the two datasets.

## 3. Data validation and EDA

`src.data.validate_measurements` enforces the six-column schema, numeric conversion, finite values, supported product variants and broad hardcoded physical ranges. Batch size is capped at 5,000 records. `load_training_data` checks target 0/1 membership and records null and duplicate counts. Reported duplicates are not silently deleted because repeated operating states can be meaningful examples. EDA notebook shows class balance, distributions, correlations, and data quality.

## 4. Preprocessing and feature engineering

`src.preprocessing.MachiningFeatures` adds three engineered numeric features inside the ML pipeline:

- `Temperature difference [K] = Process temperature − Air temperature`.
- `Spindle power [kW] = RPM × Torque × (2π/60) / 1000`.
- `Wear torque [min Nm] = Tool wear × Torque`.

`ColumnTransformer` imputes numerical medians, standardizes numeric columns, imputes categorical mode and uses one-hot encoding with unknown handling. These transformations are fitted **inside** the training folds, not on hold-out data. Input validation normally disallows missing fields; imputation is defensive in the training pipeline.

## 5. Models, calibration and splits

`random_state=42`. `train_test_split(..., stratify=y)` reserves 20% test first and splits the remaining 80% into 60% training, 20% validation (relative to the full dataset). Candidate estimators: balanced Logistic Regression, balanced pruned Decision Tree, balanced Random Forest. `CalibratedClassifierCV(estimator=pipeline, cv=3, method='sigmoid')` fits each candidate and calibrates out-of-fold training predictions.

Each candidate is evaluated on validation data. The winning candidate is selected by **highest validation average precision** (PR-AUC summary appropriate for rare positive labels), with validation ROC-AUC as tie-breaker. A classification score threshold is chosen **only on validation labels** to maximize F1 over a predefined score grid. This threshold is used on the untouched test partition for accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score and confusion matrix. The selected training-fit pipeline is serialized with joblib. Changing its training data causes retraining.

**The test set is not used in model selection, threshold selection or importance estimation.** Calibration improves score interpretability on similarly distributed data but does not prove real-world calibration. Given synthetic patterns, held-out scores cannot establish deployment performance.

## 6. Inference and dashboard

`src.predict.predict_many` validates inputs and computes actual `predict_proba` with the saved fitted estimator. `Predicted failure` is the calibrated score compared with the selected threshold. Review priority is a transparent heuristic: low / monitor / high / critical, computed from the score and threshold. It is **not a validated fault severity label or maintenance schedule**.

`local_sensitivity` changes each feature in turn to its training median or mode and observes the score difference. This is one-feature-at-a-time **what-if sensitivity**, not causal attribution or mechanical fault localization. Model Intelligence displays *permutation importance* for the six raw feature inputs measured on validation records, ranked by lost average precision when shuffled.

CSV batch predictions preserve input columns in downloads but only the required six are passed into the classifier. Individual HTML reports encode user-supplied measurement strings to prevent markup injection. History is stored in a capped local CSV in `runtime/`, ignored by Git. Exported reports require no server after download.

## 7. Key implementation files

| File | Responsibility |
|---|---|
| `bootstrap.py` | optional UCI download, fallback dataset, source/hash-aware model training |
| `launcher.py` | start Streamlit, check health, open browser, shutdown on Ctrl+C |
| `src/simulation.py` | reproducible independent mock data generator |
| `src/data.py` | input/dataset validation and metadata |
| `src/preprocessing.py` | leakage-safe feature pipeline |
| `src/modeling.py` | 3-algorithm comparison, calibration, training, metrics, serialization |
| `src/predict.py` | batch inference, priority policy, local sensitivity, HTML report |
| `src/history.py` | local capped prediction history |
| `app.py` and `ui/` | Streamlit controls, graphs, views |
| `models/*.joblib`, `*.json` | trained model and authentic results |

## 8. Dependencies and runtime

Python 3.10–3.13, latest compatible releases within `requirements.txt`: NumPy, Pandas, scikit-learn, Joblib, Streamlit, Plotly. Windows launcher needs internet the first time to install these packages. Official UCI download is optional; a pre-trained fallback simulator model is included. Model training is CPU-only with no GPU required; 8GB RAM is sufficient for this intended dataset. Docker, APIs, credentials and cloud DBs are not used.

## 9. Security and limitations

Do not unpickle untrusted `.joblib` files because Python pickle can execute code. Package's model is generated by project training; only load it from a trusted repository. Machine measurements may be sensitive if sourced from actual factories; local history should be protected appropriately. The Streamlit server binds to localhost `127.0.0.1` in the included launcher. This is not a safety-critical monitoring product, certified failure detector, online IoT system, or remaining-useful-life forecast.

## 10. Reproducibility

From the extracted project folder with dependencies installed:

```bash
python -m src.simulation
python bootstrap.py --force-train
python -m pytest -q
python -m streamlit run app.py
```

The simulator uses seed 2026; modeling uses seed 42. Exact numerical metrics may vary with dependency versions/platform. The saved metrics file is the authority for this packaged model, and the app reads it rather than displaying hardcoded values. Source code and documentation contain no fake benchmark claims.
