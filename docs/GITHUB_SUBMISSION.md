# GitHub repository submission — step-by-step

This repository is intentionally public-upload-ready. **You**, not the project ZIP, must create a real GitHub repository and paste its real link into the college spreadsheet.

## Suggested repository details

- **Repository name:** `MachineGuard-AI`
- **Description:** `Interactive ML-powered CNC machining failure-risk inspection system with batch predictions, model insights, and a Streamlit dashboard.`
- **Visibility:** Public, if permitted by your institution.
- **License:** MIT for the original project software; optional official AI4I benchmark retains CC BY 4.0.

## Option A — GitHub website (no Git commands)

1. Log in to https://github.com and click **+ → New repository**.
2. Enter `MachineGuard-AI`, add the description above, and choose Public. Since the project already includes `README.md`, `.gitignore` and `LICENSE`, do not ask GitHub to generate new ones.
3. Create the repository, then choose **uploading an existing file** (or **Add file → Upload files**).
4. From the **extracted** project folder, upload the project files and directory contents, **not the ZIP itself**. Maintain directory structure (`src/`, `ui/`, `models/`, `data/`, `tests/`, `docs/`, etc.). GitHub's browser upload may not retain empty folders; no important folders are empty.
5. **Do not upload** `.venv/`, `__pycache__/`, `runtime/`, `.pytest_cache/`, or a locally downloaded `data/ai4i2020_official.csv` unless you intend to share it under its own license and correct attribution.
6. Write a clear commit message such as `Add MachineGuard AI application and documentation` and commit changes.
7. Verify that `START_PROJECT.bat`, `app.py`, `requirements.txt`, `models/trained_model.joblib`, `models/training_results.json`, `data/machining_simulated.csv`, `README.md`, and all documentation are visible.

## Option B — Git command line

From the extracted project folder:

```bash
git init
git add .
git status
git commit -m "Initial MachineGuard AI release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MachineGuard-AI.git
git push -u origin main
```

Create an **empty** repo first and replace `YOUR_USERNAME` with your actual GitHub username. The URL above is a template, **not a claim that a real repo already exists**. If Git complains about username/email for commit, set your Git author details locally first. Authentication may require the GitHub browser flow or a supported credential manager.

## Confirm that the submission is usable

- Open your GitHub repo in a private browser tab (incognito) and verify it is publicly readable.
- Click `README.md` and check that its relative links resolve.
- Check `models/trained_model.joblib` is in the repo and no secrets/private machine data are committed.
- Prefer a local clean-clone test in Windows: download the repo ZIP and launch `START_PROJECT.bat`.
- Verify the documented source provenance matches the dataset used for the saved model. The model is retrained automatically if the official UCI dataset is downloaded later.

## Submit in the college Excel sheet

1. Open the college-provided Excel spreadsheet.
2. Find your corresponding row and the column labeled **GitHub Link**, **Repository URL** or equivalent.
3. Copy the real URL from your repository page, for example `https://github.com/YOUR_USERNAME/MachineGuard-AI` (substitute your own actual URL).
4. Paste it as a hyperlink or text into the designated cell.
5. Save and re-open the sheet to verify the complete URL is present and accessible.

## Academic integrity checklist

Use the project only in accordance with your institution's rules for AI-assisted submissions. Inspect and understand the code and reported metrics before presenting it; preserve dataset citations. Do not claim simulator results are industrial measurements or that you carried out Windows-specific tests you did not actually run.
