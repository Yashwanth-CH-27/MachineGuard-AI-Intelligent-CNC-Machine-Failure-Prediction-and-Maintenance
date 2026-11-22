# One-minute launch and demonstration checklist

### Run the app

1. Install Python 3.10–3.13 with PATH checked and connect to the internet for first-run dependency installation.
2. Extract the project ZIP, open the extracted folder, and double-click **START_PROJECT.bat**.
3. The local browser app opens at **http://localhost:8501** once the server is ready. Keep the command window open.
4. Look at the source indicator. It will explicitly say either **locally simulated demo** or **UCI AI4I published synthetic**; neither is an industrial real-time machine feed.

### Show the lecturer

1. **Overview:** point out number of records, labeled failure prevalence, and a real held-out test metric.
2. **Check a Machine:** enter L, 302.0 K, 309.0 K, 1300 RPM, 42 Nm, 170 min. Run inspection. Explain the calculated risk score and saved heuristic priority. Download the HTML report.
3. **Batch Inspection:** download the sample input, upload it and run CSV predictions. Download the results CSV.
4. **Model Intelligence:** compare Logistic Regression, Decision Tree and Random Forest; discuss why the minority failure class makes recall and average precision important. Show the confusion matrix and permutation importance.
5. **Project Guide:** explain *what the project cannot do*: no live sensors, no certified fault diagnosis, and no accurate remaining-life estimate.

### Stop

Focus the launcher Command Prompt and press **Ctrl+C**. For additional detail use `README.md`, `docs/TECHNICAL_DOCUMENTATION.md`, and `docs/VIVA_QUESTIONS.md`.
