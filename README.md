# Titanic ML Web App (Flask)

This is a simple machine learning web app that trains a Random Forest classifier on the Titanic dataset and serves:

- A web UI to enter passenger details and get a survival prediction + confidence
- A model accuracy score
- A feature-importance bar chart (Chart.js)

## Run locally (Windows / PowerShell)

```powershell
cd "c:\Users\a\Downloads\cursor example project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes

- The app tries to load the Titanic dataset from `seaborn.load_dataset("titanic")`.
- If that fails (offline), it will fall back to `data/titanic.csv` if you provide it.

