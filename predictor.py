import unicodedata
from pathlib import Path

import fastf1
import joblib
import pandas as pd

# All paths are built from this file's location, so the app works no matter where it is started from.
BASE_DIR = Path(__file__).resolve().parent
CURRENT_SEASON = 2026

# FastF1 cache: the first download of a session is slow, later ones come from this folder.
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# CIRCUIT MAPPING

RACE_TO_CIRCUIT = {
    "Australia": "albert_park", "China": "shanghai", "Japan": "suzuka",
    "Bahrain": "bahrain", "Saudi Arabia": "jeddah", "Miami": "miami",
    "Monaco": "monaco", "Spain": "catalunya", "Canada": "villeneuve",
    "Austria": "red_bull_ring", "Britain": "silverstone", "Belgium": "spa",
    "Hungary": "hungaroring", "Netherlands": "zandvoort", "Italy": "monza",
    "Azerbaijan": "baku", "Singapore": "singapore", "USA": "cota",
    "Mexico": "rodriguez", "Brazil": "interlagos", "Las Vegas": "las_vegas",
    "Qatar": "losail", "Abu Dhabi": "yas_marina",
}

# The Kaggle data may spell a few circuits differently. Used only if the first spelling is not in the encoder.
CIRCUIT_ALTERNATIVES = {
    "cota": "americas",
    "singapore": "marina_bay",
    "las_vegas": "vegas",
}

# NAME MATCHING
# Driver and team names are written differently in different sources
# (FastF1: "Max Verstappen" / "Red Bull Racing", training data: "Verstappen" / "Red Bull").


def _norm(text):
    """lower-case, no accents, no extra spaces:  'Hülkenberg ' -> 'hulkenberg'"""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower()


RB_TEAM = ["RB F1 Team", "RB", "AlphaTauri", "Toro Rosso"]
TEAM_ALIASES = {
    "red bull racing": ["Red Bull"],
    "racing bulls": RB_TEAM,
    "rb": RB_TEAM,
    "kick sauber": ["Sauber", "Alfa Romeo"],
}


def resolve_name(name, known):
    """Return the spelling used in `known` that matches `name`. If nothing matches, return the cleaned name."""
    clean = str(name).strip()
    lookup = {_norm(k): k for k in known}
    key = _norm(clean)

    if key in lookup:
        return lookup[key]
    for alias in TEAM_ALIASES.get(key, []):
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    if len(key) >= 4:
        for k_norm, k in lookup.items():
            if key in k_norm or k_norm in key:
                return k
    return clean


def to_surname(full_name, known):
    """'Max Verstappen' -> 'Verstappen' (the format used in the training data). Surnames pass through unchanged."""
    parts = str(full_name).strip().split()
    lookup = {_norm(k): k for k in known}
    for n in (2, 1):
        candidate = _norm(" ".join(parts[-n:]))
        if candidate in lookup:
            return lookup[candidate]
    return parts[-1] if parts else str(full_name)


# LOAD ARTIFACTS

class ModelBundle:
    """Hold everything loaded once, reused across requests."""
    def __init__(self):
        models = BASE_DIR / "models"
        self.model = joblib.load(models / "f1_xgboost_model.pkl")
        self.features = joblib.load(models / "model_features.pkl")
        self.le_circuit = joblib.load(models / "le_circuit.pkl")
        self.le_constructor = joblib.load(models / "le_constructor.pkl")
        self.le_driver = joblib.load(models / "le_driver.pkl")

        self._check_circuit_ids()
        self.history_df, self._round_lookup = self._load_history()

    def _check_circuit_ids(self):
        """Make sure every circuit id in RACE_TO_CIRCUIT is one the encoder knows."""
        known = set(self.le_circuit.classes_)
        for race, circuit in list(RACE_TO_CIRCUIT.items()):
            if circuit in known:
                continue
            alternative = CIRCUIT_ALTERNATIVES.get(circuit)
            if alternative in known:
                RACE_TO_CIRCUIT[race] = alternative
            else:
                print(f"Warning: circuit '{circuit}' ({race}) is not in the saved encoder.")

    def _load_history(self):
        """2018-2024 training history plus the 2026 races downloaded with FastF1."""
        keep = ["year", "round", "driver_name", "constructor_name", "finish_position", "points", "circuit_id"]
        base = pd.read_csv(BASE_DIR / "data" / "processed" / "f1_eda_ready.csv")
        history = base[keep].copy()

        # (year, circuit) -> round number; the earliest round is kept if a circuit hosts twice (2020)
        round_lookup = {}
        for year, rnd, circuit in base[["year", "round", "circuit_id"]].drop_duplicates().itertuples(index=False):
            round_lookup.setdefault((int(year), circuit), int(rnd))

        try:
            f26 = pd.read_csv(BASE_DIR / "data" / "processed" / "f1_2026_results.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            f26 = pd.DataFrame()

        if len(f26) > 0:
            drivers = set(history["driver_name"].unique())
            teams = set(history["constructor_name"].unique())

            f26["driver_name"] = f26["driver_name"].apply(lambda n: to_surname(n, drivers))
            f26["constructor_name"] = f26["constructor_name"].apply(lambda n: resolve_name(n, teams))

            race_order = list(dict.fromkeys(f26["race_name"]))
            round_of = {name: i + 1 for i, name in enumerate(race_order)}
            f26["round"] = f26["race_name"].map(round_of)
            f26["year"] = CURRENT_SEASON
            f26["finish_position"] = pd.to_numeric(f26["finish_position"], errors="coerce").fillna(20)
            f26["points"] = pd.to_numeric(f26["points"], errors="coerce").fillna(0)
            f26["circuit_id"] = f26["race_name"].map(RACE_TO_CIRCUIT)

            for name, rnd in round_of.items():
                circuit = RACE_TO_CIRCUIT.get(name)
                if circuit:
                    round_lookup.setdefault((CURRENT_SEASON, circuit), rnd)

            history = pd.concat([history, f26[keep]], ignore_index=True)

        history = history.sort_values(["year", "round"]).reset_index(drop=True)
        return history, round_lookup

    def history_before(self, season, race_name):
        """Only races that happened BEFORE the race we want to predict, so the answer is never in the history."""
        season = int(season)
        circuit = RACE_TO_CIRCUIT.get(race_name)
        rnd = self._round_lookup.get((season, circuit))
        h = self.history_df

        if rnd is None:
            # race not in our data: a future race (use everything we have) or a season we have no record of
            if season >= h["year"].max():
                return h[h["year"] <= season]
            return h[h["year"] < season]
        return h[(h["year"] < season) | ((h["year"] == season) & (h["round"] < rnd))]


# QUALIFYING FETCHER

def load_qualifying(season: int, race_name: str) -> pd.DataFrame:
    session = fastf1.get_session(season, race_name, "Q")
    # Only the results table is needed, so skip laps, telemetry, weather and messages (much faster).
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results.copy()

    position = pd.to_numeric(results["Position"], errors="coerce")
    # a driver without a qualifying time has no position: put them at the back
    fill_value = position.max() + 1 if position.notna().any() else 20
    position = position.fillna(fill_value).astype(int)

    return pd.DataFrame({
        "driver_name": results["FullName"].values,
        "constructor_name": results["TeamName"].values,
        "grid_position": position.values,   # qualifying order (grid penalties are not included)
        "quali_position": position.values,
    })


# FEATURE BUILDER

def build_race_features(quali_df, history_df, circuit_id, le_driver, le_constructor, le_circuit,
                        season=CURRENT_SEASON):
    """
    Same features as in training.
      history_df : only races BEFORE the one being predicted (use bundle.history_before)
      season     : year of the race, used for "points so far THIS season"
    """
    history_df = history_df.sort_values(["year", "round"])
    known_drivers = set(history_df["driver_name"].dropna().unique())
    known_teams = set(history_df["constructor_name"].dropna().unique())

    enc_drivers = set(le_driver.classes_)
    enc_teams = set(le_constructor.classes_)
    enc_circuits = set(le_circuit.classes_)

    rows = []
    for _, row in quali_df.iterrows():
        driver = to_surname(row["driver_name"], known_drivers)
        constructor = resolve_name(row["constructor_name"], known_teams)

        driver_hist = history_df[history_df["driver_name"] == driver]
        constructor_hist = history_df[history_df["constructor_name"] == constructor]
        driver_season = driver_hist[driver_hist["year"] == season]
        constructor_season = constructor_hist[constructor_hist["year"] == season]

        # Recent form: last 3 races
        avg_finish_last3 = driver_hist["finish_position"].tail(3).mean()
        avg_points_last3 = driver_hist["points"].tail(3).mean()
        constructor_avg_points_last3 = constructor_hist.groupby(["year", "round"])["points"].sum().tail(3).mean()

        # Reliability: last 5 races
        dnf_rate = (driver_hist["finish_position"].tail(5) > 15).mean() if len(driver_hist) > 0 else 0

        # Record at this circuit
        circuit_hist = driver_hist[driver_hist["circuit_id"] == circuit_id]
        win_rate = (circuit_hist["finish_position"] == 1).mean() if len(circuit_hist) > 0 else 0

        # Championship points THIS season only
        driver_points = driver_season["points"].sum()
        constructor_points = constructor_season["points"].sum()

        # -1 = the saved encoder has never seen this name (new drivers and teams)
        driver_encoded = le_driver.transform([driver])[0] if driver in enc_drivers else -1
        constructor_encoded = le_constructor.transform([constructor])[0] if constructor in enc_teams else -1
        circuit_encoded = le_circuit.transform([circuit_id])[0] if circuit_id in enc_circuits else -1

        rows.append({
            "driver_name": row["driver_name"],
            "constructor_name": row["constructor_name"],
            "grid_position": row["grid_position"],
            "quali_position": row["quali_position"],
            "driver_avg_finish_last3": avg_finish_last3,
            "driver_avg_points_last3": avg_points_last3,
            "constructor_avg_points_last3": constructor_avg_points_last3,
            "driver_win_rate_at_circuit": win_rate,
            "driver_dnf_rate": dnf_rate,
            "driver_points_so_far": driver_points,
            "constructor_points_so_far": constructor_points,
            "driver_encoded": driver_encoded,
            "constructor_encoded": constructor_encoded,
            "circuit_encoded": circuit_encoded,
        })

    pred_df = pd.DataFrame(rows)
    pred_df["driver_avg_finish_last3"] = pred_df["driver_avg_finish_last3"].fillna(20)
    remaining_cols = [
        "driver_avg_points_last3", "constructor_avg_points_last3",
        "driver_win_rate_at_circuit", "driver_dnf_rate",
        "driver_points_so_far", "constructor_points_so_far",
    ]
    pred_df[remaining_cols] = pred_df[remaining_cols].fillna(0)
    return pred_df


# PREDICTOR

def predict_race(season: int, race_name: str, bundle: ModelBundle) -> pd.DataFrame:
    if race_name not in RACE_TO_CIRCUIT:
        raise ValueError(f"Unknown race: {race_name}")
    circuit_id = RACE_TO_CIRCUIT[race_name]
    quali_df = load_qualifying(season, race_name)
    history = bundle.history_before(season, race_name)
    feature_df = build_race_features(
        quali_df, history, circuit_id,
        bundle.le_driver, bundle.le_constructor, bundle.le_circuit,
        season=season,
    )
    X_pred = feature_df[bundle.features]
    raw = bundle.model.predict_proba(X_pred)[:, 1]
    # scale so the whole race adds up to 100% (only one driver can win)
    feature_df["win_probability"] = raw / raw.sum() if raw.sum() > 0 else raw

    return feature_df[["driver_name", "constructor_name", "grid_position", "win_probability"]] \
           .sort_values(by="win_probability", ascending=False)
