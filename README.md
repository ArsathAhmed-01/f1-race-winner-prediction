# F1 Race Winner Prediction

A machine learning project that estimates each driver's chance of winning a Formula 1 race from qualifying results and recent form. It uses historical race data (2018-2024) plus 2026 results downloaded with FastF1, and an XGBoost classifier.

> **Credit:** this project is built on top of [Parthshewale18/F1](https://github.com/Parthshewale18/F1) (MIT License). The original project structure, the Streamlit app and the API come from that repository. See [What I changed](#what-i-changed) for my own work.

---

## How it works

Each driver in a race is one row. The model predicts the probability that this driver wins, and the probabilities are scaled so the whole race adds up to 100%.

**Features (12)**

| Group | Features |
|---|---|
| Starting position | grid position, qualifying position |
| Driver form | average finish and average points over the last 3 races, DNF rate over the last 5 races |
| Team form | team average points over the last 3 races |
| Circuit | driver's win rate at this circuit |
| Championship | driver and team points so far **this season** |
| Encoded names | driver, team and circuit (label encoded) |

All rolling features use `shift(1)`, so a race only sees information from *before* that race (no data leakage).

## Project pipeline

Run the notebooks in this order:

| # | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/data_collection.ipynb` | Loads the Kaggle F1 data, keeps 2018+, merges tables, downloads 2026 results with FastF1 |
| 2 | `notebooks/eda.ipynb` | Exploratory analysis (class imbalance, grid vs wins, DNFs, constructor dominance) |
| 3 | `notebooks/feature_engineering.ipynb` | Builds the rolling features and saves the encoders |
| 4 | `notebooks/model_training.ipynb` | Trains XGBoost with GroupKFold cross-validation and saves the model |
| 5 | `notebooks/predict.ipynb` | Predicts a race and tests it on a completed race |

## Results

Only about 5% of rows are winners, so plain accuracy is misleading (always predicting "nobody wins" is already about 95% accurate). The main metric is **Top-3 accuracy**: how often the real winner is among the model's 3 most likely drivers.

Cross-validation uses `GroupKFold` (5 folds) so the same race never appears in both training and validation.

| Metric (average of 5 folds) | Value |
|---|---|
| Top-3 accuracy | **XX%** |
| Log loss | **X.XX** |

## Setup

Python 3.12 is recommended (newer versions may not have prebuilt packages for pandas).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install fastf1 pandas numpy matplotlib seaborn scikit-learn xgboost plotly joblib ipykernel jupyter
```

**Data:** download the Kaggle dataset *"Formula 1 World Championship (1950 - 2020)"* and put the CSV files (`races.csv`, `results.csv`, `drivers.csv`, `constructors.csv`, `qualifying.csv`, `circuits.csv`, ...) in `data/raw/`. These files are not stored in this repository because of their size.

Then open the notebooks in VS Code, select the `.venv` kernel, and run them in the order above.

## Repository structure

```
F1/
├── notebooks/        # the 5 notebooks of the pipeline
├── models/           # trained model and encoders (.pkl)
├── data/
│   ├── raw/          # Kaggle CSV files (not in git)
│   └── processed/    # datasets created by the notebooks
├── app.py            # Streamlit app (from the original project)
├── api.py            # FastAPI service (from the original project)
├── predictor.py      # prediction logic used by the app
├── schemas.py
└── requirements.txt
```

## What I changed

- Re-ran the full pipeline in a clean Python 3.12 environment.
- Fixed the cross-validation log loss, which was calculated from 0/1 predictions instead of probabilities.
- Rebuilt `predict.ipynb`:
  - automatic matching of driver and team names (accents, team renames such as *Red Bull Racing* / *Red Bull*)
  - "points so far" and DNF rate now match how they were computed in training (current season / last 5 races)
  - 2026 results are added so recent form is up to date, and a test race only uses earlier races
  - probabilities are normalised so each race adds up to 100%

## Limitations

- The Kaggle data ends in 2024, so the model was **not trained** on 2025-2026 seasons. New drivers and teams (for example Audi, Cadillac) are unknown to the saved encoders and get the value -1.
- The DNF flag is an approximation (`finish_position > 15`), not the real race status.
- The model needs qualifying results, so it cannot predict before qualifying.
- Testing on a single race says very little. Use the cross-validation numbers as the real measure.

## Future improvements

- Retrain with 2025 and 2026 data and refit the encoders.
- Build the DNF flag from the race status.
- Compare against simple baselines (pole position wins).
- Check that the Streamlit app uses the same feature logic as `predict.ipynb`.

## License

MIT, see the [LICENSE](LICENSE) file. The original copyright notice is kept as required by the license.
