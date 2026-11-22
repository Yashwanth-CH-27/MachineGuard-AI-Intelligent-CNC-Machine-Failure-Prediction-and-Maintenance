# Dataset attribution and provenance

## Packaged data — locally simulated, NOT UCI AI4I

`data/machining_simulated.csv` contains **10,000 independently generated illustrative machining records** created with `src/simulation.py` (random seed 2026). This is a custom simulator dataset, **not** the original AI4I 2020 benchmark. The simulator's variables and example failure mechanisms were inspired by the publicly documented AI4I 2020 research problem; its numerical records and failure labels were generated here, not downloaded from UCI or measured on a factory machine. Reported metrics on this dataset are not official benchmark scores.

## Optional official public dataset — AI4I 2020

When internet access is available on the first Windows launch, `bootstrap.py --prefer-official` tries to retrieve the **original** public dataset and saves it as `data/ai4i2020_official.csv`. It verifies 10,000 records and the required feature/target names, and retrains the model on that original synthetic benchmark. If it cannot be downloaded, the included simulator and pretrained simulator model continue to work. For a manual retry, run `python bootstrap.py --refresh-official` with the project's Python environment active (or use `.venv\Scripts\python.exe bootstrap.py --refresh-official` on Windows).

- Dataset: **AI4I 2020 Predictive Maintenance Dataset**
- Creator / introductory paper: **Stephan Matzka (2020), Explainable Artificial Intelligence for Predictive Maintenance Applications**, IEEE AI4I 2020.
- Host: **UCI Machine Learning Repository**
- Dataset DOI: **10.24432/C5HS5C**
- Official webpage: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Direct CSV: https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)** https://creativecommons.org/licenses/by/4.0/
- UCI describes the 10,000 records as **synthetic** and designed to resemble industrial maintenance observations.

## Inputs and labels

The model consumes exactly: product variant `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.

The label is `Machine failure` (0/1). Original AI4I identifiers (`UDI`, `Product ID`) and mode/target indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are never used as predictive inputs. Predicting using those labels would leak the answer.

The product variant indicates product quality, **not a particular CNC machine model**.

## License distinctions

The original code and independently generated simulated data distributed in this repository are published under its MIT license. The external UCI AI4I dataset is **CC BY 4.0** and remains governed by the dataset creator's license. The default `.gitignore` excludes any optional locally downloaded copy to prevent accidental source confusion; if you share it separately, retain this attribution and indicate changes.
