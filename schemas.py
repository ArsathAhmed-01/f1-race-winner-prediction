from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    season : int = Field(..., ge=2018, le=2030, examples=[2026], description="The season year for the race")
    race_name : str = Field(..., examples=["Monaco"])

class DriverPrediction(BaseModel):
    driver_name: str
    constructor_name: str
    grid_position: int
    win_probability: float

class PredictResponse(BaseModel):
    winner: str
    winner_constructor: str
    win_probability: float
    rankings: List[DriverPrediction]