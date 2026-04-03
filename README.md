# Titanic ML Web App (Flask)

This is a simple machine learning web app that trains a Random Forest classifier on the Titanic dataset and serves:

- A web UI to enter passenger details and get a survival prediction + confidence
- A model accuracy score
- A feature-importance bar chart (Chart.js)

Features

- **Live predictions** — Input passenger details and get instant survival prediction with confidence score
- **Probability breakdown** — See both survive % and perish % for every prediction
- **Feature importance chart** — Interactive horizontal bar chart (Chart.js) showing which features matter most
- **Model accuracy badge** — Displayed prominently, computed on a held-out test split
- **Clean dark UI** — Black/greyscale-white theme with noise texture and grid overlay
- **Zero external DB** — Dataset is bundled as a CSV; model trains at startup

### Model

- **Algorithm:** Random Forest Classifier (`sklearn`)
- **Trees:** 200 estimators
- **Features:** `pclass`, `sex`, `age`, `fare`, `embarked`
- **Target:** `survived` (0 = perished, 1 = survived)
- **Accuracy:** ~74% on the held-out test set

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Renders the frontend page |
| `POST` | `/predict` | Returns prediction JSON |


## Notes

- The app tries to load the Titanic dataset from `seaborn.load_dataset("titanic")`.
- If that fails (offline), it will fall back to `data/titanic.csv` if you provide it.

