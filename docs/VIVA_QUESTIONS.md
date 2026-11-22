# 32 viva questions and student-friendly answers

**1. What is the main aim of MachineGuard AI?** To estimate whether CNC-style machining conditions resemble failure-labeled records and present inspection-oriented results in a user-friendly application.

**2. What type of machine are you studying?** A generic industrial machining process representative of CNC-style operations. The dataset does not identify specific CNC makes, models or individual machines.

**3. What measurements does the model use?** Product quality variant L/M/H, air temperature, process temperature, spindle RPM, torque, and tool wear.

**4. What does L/M/H indicate?** The quality category of the produced item, **not** machine model or machine health.

**5. Why is this a machine learning project?** We fit a classifier using labeled example measurements; output probabilities are derived from the learned fitted model, not just a set of hardcoded UI conditions.

**6. What are the two possible labels?** `0` means no failure label for the record; `1` means a machine failure label.

**7. Is the training dataset collected from real factories?** No. The included 10,000-row dataset is independently generated for an offline demo. The optional original UCI AI4I 2020 benchmark is also synthetic.

**8. What is the official dataset source?** UCI AI4I 2020 Predictive Maintenance Dataset, DOI 10.24432/C5HS5C, licensed CC BY 4.0.

**9. Why include simulator data?** The application can train and demonstrate its workflow without depending on external dataset downloading. Its separate provenance is clearly labeled.

**10. Why is dataset attribution necessary?** It credits research and external data creators and prevents incorrect claims about experiments or measurement provenance.

**11. What is supervised learning?** Learning a mapping from input features to known target labels.

**12. Is the task classification or regression?** Binary classification because the target is machine failure 0 or 1; a score is also produced.

**13. Which models were compared?** Logistic Regression, Decision Tree, and Random Forest.

**14. Why Random Forest?** It can learn interactions and nonlinear boundaries between speed, torque, temperature and tool wear. The final model still depends on validation scores; it is not selected blindly.

**15. Why have a Logistic Regression baseline?** It gives a simple linear benchmark and helps show whether complex models add value.

**16. What is preprocessing?** Converting raw inputs into model-ready features, handling missing values defensively, scaling numbers and one-hot encoding the categorical type.

**17. What feature engineering did you perform?** Temperature difference, approximate spindle power based on speed and torque, and wear multiplied by torque.

**18. What is data leakage?** When training uses information unavailable at prediction time, such as a failure-mode target flag, or learns preprocessing statistics from held-out data. Both would inflate evaluation.

**19. Which columns are deliberately excluded?** UDI, Product ID, and failure-mode labels TWF, HDF, PWF, OSF, RNF. Only the six real input measurements are used.

**20. Why split the dataset into three parts?** Train fits candidate models, validation selects model and score threshold, and the untouched test set evaluates the final selection.

**21. What is a stratified split?** A split that approximately preserves the failure/nonfailure ratio within each subset.

**22. Why is class imbalance important?** Failure labels are uncommon, so a model that predicts only normal could appear accurate while missing failures.

**23. Define precision.** Of records the model flags as failure, the fraction that actually have the failure label.

**24. Define recall.** Of records actually labeled as failure, the fraction flagged by the model.

**25. What is F1?** Harmonic mean of precision and recall, balancing the two for a chosen classification threshold.

**26. What is ROC-AUC?** A threshold-independent ranking summary of true-positive versus false-positive rates across score cutoffs.

**27. Why select using average precision?** It summarizes precision-recall ranking behavior, often more informative than accuracy when failures are rare.

**28. What is the confusion matrix?** A 2×2 comparison of predicted vs actual normal and failure labels: true negatives, false positives, false negatives and true positives.

**29. What does calibration mean?** Adjusting model scores using training-fold held-out predictions so they better reflect observed positive-label frequencies in similar data. This does not guarantee industrial probability accuracy.

**30. Can this predict the exact next failure time?** No. It predicts labels/risk scores for provided measurements, not remaining useful life or a future timestamp.

**31. Does the app identify a precise fault such as a broken bearing?** No. It predicts general failure risk. Local sensitivity and permutation importance show model reliance, not confirmed mechanical causation.

**32. What makes your software demonstrable?** It includes a Windows launcher, offline-capable saved model, manual and CSV prediction, validation, real model evaluation plots, local history, downloadable reports, and public-repo-ready documentation.

## Extra questions lecturers may ask

**What is overfitting?** When a model memorizes training-specific patterns and performs worse on new records. Held-out validation/test checks and tree regularization are used to assess it.

**What is underfitting?** When a model is too simple to represent useful patterns, leading to poor train and validation performance.

**What is the purpose of a decision threshold?** The score is continuous; a threshold turns it into a binary class. We choose it using validation F1, not the test set.

**How are maintenance priorities produced?** They are clearly labeled score-based **heuristic** review categories, not a separately trained severity or service scheduling model.

**How would you improve it in the real world?** Collect real sensor and maintenance logs, validate on new machines/time periods, assess missed failures and false alarms, monitor drift, and involve qualified safety/maintenance engineers.
