from fastapi import FastAPI, HTTPException
import predictor
from schemas import PredictRequest, DriverPrediction, PredictResponse

app = FastAPI(
    title="F1 Race Winner Predictor API",
    description="Predicts F1 race winner probabilities from qualifying results.",
    version="1.0.0",
)

bundle : predictor.ModelBundle | None = None

@app.on_event("startup")
def load_model():
    global bundle
    try:
        bundle = predictor.ModelBundle()
    except Exception as e:
        print(f"Error loading model: {e}")
        bundle = None

@app.get("/")
def root():
    return {
        "message": "F1 Race Winner Predictor API is running. Visit /docs to try it out."
        }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bundle is not None
    }

@app.get("/races")
def get_races():
    return list(predictor.RACE_TO_CIRCUIT.keys())

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if bundle is None:
        raise HTTPException(
            status_code = 503,
            detail = "Model is not loaded. Check the server log for the loading error."
        )
    if request.race_name not in predictor.RACE_TO_CIRCUIT:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown race : {request.race_name}"
        )
    try:
        predictions = predictor.predict_race(request.season, request.race_name, bundle)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )
    winner = predictions.iloc[0]
    return PredictResponse(
        winner = winner["driver_name"],
        winner_constructor = winner["constructor_name"],
        win_probability = float(winner["win_probability"]),
        rankings = [DriverPrediction(**row) for row in predictions.to_dict(orient="records")]
    )
    