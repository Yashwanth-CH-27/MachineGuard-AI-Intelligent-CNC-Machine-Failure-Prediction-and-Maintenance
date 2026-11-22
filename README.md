
# MachineGuard AI
### Intelligent CNC Machine Failure Prediction and Maintenance

MachineGuard AI is a local machine-learning application that evaluates industrial machining measurements and estimates whether an operating record is associated with a machine-failure label. It combines supervised classification with an interactive dashboard for individual inspections, batch analysis, and model evaluation.

The application is built with **Python, scikit-learn, Streamlit, Pandas, and Plotly**. It runs locally and does not require a cloud account, API key, or GPU.

> **Dataset scope:** The included trained model uses a separately generated **synthetic machining dataset**, not readings collected from a working CNC machine. On first launch, the application attempts to obtain the public **UCI AI4I 2020 synthetic benchmark** and retrains if the download succeeds. Predictions are educational risk estimates, not certified equipment diagnostics.

## Features

- **Dashboard:** Review dataset characteristics, failure-label distribution, operating measurements, and local inspection history.
- **Single-machine inspection:** Enter measured operating values to obtain a predicted class, estimated failure score, and suggested inspection priority.
- **Batch prediction:** Upload a CSV containing multiple observations, inspect the results, and export the predictions.
- **Model insights:** Compare fitted algorithms, examine held-out performance metrics, review a confusion matrix, and explore feature importance.
- **Reports:** Download a single-inspection HTML report or a batch-results CSV.
- **Local execution:** Start the Streamlit interface with a Windows launcher or standard Python commands.

## Operating Inputs

| Input | Description |
|---|---|
| Product type (`L`, `M`, `H`) | Quality variant of the manufactured product |
| Air temperature (K) | Surrounding air temperature |
| Process temperature (K) | Temperature of the machining process |
| Rotational speed (rpm) | Rotational speed of the tool or spindle |
| Torque (Nm) | Rotational torque |
| Tool wear (min) | Accumulated cutting-tool use |

The classification target is **Machine failure** (0 = normal label, 1 = failure label).

The pipeline additionally derives temperature difference, approximate spindle power, and a wear–torque interaction. Identifiers and failure-mode target columns are excluded from model inputs to avoid target leakage.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| User interface | Streamlit |
| Data visualization | Plotly, Matplotlib |
| Model serialization | Joblib |
| Testing | Pytest |
| Version control | Git, GitHub |

## Installation and Setup

### Prerequisites

- Python 3.10–3.13
- Windows, Linux, or macOS
- Internet connection for initial dependency installation

### Windows Quick Start

1. Clone or download the repository.
2. Extract the project if downloaded as a ZIP.
3. Open the project folder.
4. Double-click `START_PROJECT.bat`.
5. Wait for the application to initialize.
6. Open `http://localhost:8501` in your browser if it does not open automatically.

The launcher prepares the Python environment, installs the required dependencies, initializes the dataset and trained model, and starts the Streamlit server.

### Manual Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Initialize the dataset and model:

```bash
python bootstrap.py --prefer-official
```

Start the application:

```bash
python -m streamlit run app.py
```

The application will be available at:

`http://localhost:8501`

## Machine Learning Workflow

The system implements a supervised machine learning pipeline for binary machine-failure classification.

### 1. Data Collection and Validation

The project uses simulated machining measurements and supports the publicly available AI4I 2020 Predictive Maintenance Dataset.

Data validation checks the required features, input types, missing values, and invalid operating measurements.

### 2. Data Preprocessing

Preprocessing includes:

- Input validation
- Handling missing values
- Categorical feature encoding
- Numerical feature scaling where appropriate
- Feature engineering
- Stratified data splitting

### 3. Feature Engineering

Additional features include:

- Temperature difference
- Approximate rotational power
- Tool wear and torque interaction

These features help represent relationships between different operating measurements.

### 4. Model Training

Three machine learning algorithms are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The dataset is divided into training, validation, and testing partitions using a 60:20:20 ratio.

Model selection uses validation average precision. The classification threshold is selected using validation F1-score.

### 5. Model Evaluation

The trained models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Confusion matrix

### Recorded Model Performance

The included Random Forest model was trained on 10,000 simulated machining records, including 444 failure-labeled records.

| Evaluation Metric | Result |
|---|---|
| Accuracy | 98.35% |
| Precision | 91.18% |
| Recall | 69.66% |
| F1-score | 78.98% |
| ROC-AUC | 96.47% |
| Average Precision | 88.07% |

**Note:** These evaluation results are obtained from the included simulated dataset. They do not represent verified performance on real CNC machines or the original UCI benchmark. Results may change when the model is retrained on another dataset.

Detailed evaluation records are available in `models/training_results.json`.

## Application Modules

### Dashboard

The dashboard provides an overview of operating measurements and machine-failure data.

Features include:

- Dataset statistics
- Machine condition summaries
- Failure distribution charts
- Operating parameter visualizations
- Prediction history

### Machine Failure Prediction

Users can manually enter CNC machining parameters and obtain:

- Predicted machine condition
- Estimated failure-risk score
- Maintenance inspection priority
- Contributing operating factors
- Downloadable inspection report

### Batch Prediction

The batch prediction module supports uploading CSV files containing multiple machine observations.

Features include:

- CSV file validation
- Multiple-record prediction
- Prediction result tables
- Downloadable CSV reports

### Model Insights

The model insights section provides information about:

- Machine learning algorithms
- Model comparison
- Evaluation metrics
- Confusion matrix
- Feature importance
- Model limitations

## Dataset Information

### Included Dataset

The project includes a reproducibly generated synthetic machining dataset containing 10,000 observations.

The dataset is generated through the project's simulation module.

### Public Reference Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

**DOI:** https://doi.org/10.24432/C5HS5C

**License:** CC BY 4.0

The public dataset contains simulated industrial machining conditions and machine-failure labels.

No physical sensor integration or live CNC machine monitoring is included in the current implementation.

## Project Structure

```text
MachineGuard-AI/
|
|-- app.py
|-- START_PROJECT.bat
|-- bootstrap.py
|-- launcher.py
|-- requirements.txt
|-- requirements-dev.txt
|-- README.md
|
|-- assets/
|-- data/
|-- models/
|-- src/
|-- ui/
|-- notebooks/
|-- tests/
|-- docs/
```

### Directory Description

- `app.py` — Main Streamlit application
- `START_PROJECT.bat` — Windows application launcher
- `bootstrap.py` — Dataset and model initialization
- `launcher.py` — Local application startup
- `assets/` — User interface styling
- `data/` — Datasets and sample inputs
- `models/` — Trained models and evaluation records
- `src/` — Machine learning pipeline and prediction logic
- `ui/` — Streamlit interface components
- `notebooks/` — Exploratory data analysis
- `tests/` — Automated testing
- `docs/` — Technical documentation

## Testing

The project includes automated tests for:

- Dataset validation
- Data preprocessing
- Feature engineering
- Model predictions
- CSV input handling
- Report generation
- Model training

Run the tests using:

```bash
python -m pip install -r requirements-dev.txt
python bootstrap.py
python -m pytest -q
```

## Limitations

- The system is designed for simulated CNC-style machining conditions.
- The model is not validated for a particular CNC manufacturer or equipment model.
- Failure predictions do not identify confirmed mechanical defects.
- The application does not predict exact remaining machine lifetime.
- Maintenance priorities are advisory and should not replace professional inspection.
- Model performance on simulated data does not establish real industrial accuracy.
- Continuous IoT sensor monitoring is not implemented.

## Future Enhancements

- Integration with IoT sensors for continuous monitoring
- Model evaluation on real industrial machine data
- Time-series analysis of equipment operating conditions
- Remaining useful life estimation
- Maintenance scheduling and alert notifications
- Advanced failure analysis and explainable AI
- Cloud-based monitoring across multiple machines

## Project Team

**Batch 7 — KL University**

**Course:** AI/ML with Applied Mathematics

| Team Member | Roll Number |
|---|---|
| Chundru Yaswanth Sai Kiran | 2601050019 |
| Bhanu Teja Tiruveedula | 2601150011 |
| P. Sri Krishna | 2601150099 |
| Guttikonda Sudheer | 2601050036 |

**Project Supervisor:** Prapulla Kumar Rondla

## References

1. UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.
2. Scikit-learn Documentation — https://scikit-learn.org/
3. Streamlit Documentation — https://docs.streamlit.io/
4. Pandas Documentation — https://pandas.pydata.org/
5. NumPy Documentation — https://numpy.org/

## License

The application code is provided under the MIT License.

The referenced UCI AI4I dataset is separately licensed under Creative Commons Attribution 4.0 International.
