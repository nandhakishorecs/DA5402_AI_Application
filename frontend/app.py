from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np

# Import the CricketTargetScorePredictor class from src/model.py
from model1 import CricketTargetScorePredictor

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class PredictionInput(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    venue: str
    inning: int
    total_runs: int
    is_wicket: int
    over: float
    team1_games_played: int
    team1_games_won: int
    team2_games_played: int
    team2_games_won: int

# Load the predictor model
predictor = CricketTargetScorePredictor()
MODEL_PATH = "../models/target_score_model_best.pth"
PREPROCESS_PATH = "../models/preprocess_best.joblib"

if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESS_PATH)):
    raise FileNotFoundError("Model or preprocess files not found.")

predictor.load_model(MODEL_PATH, PREPROCESS_PATH)

# Endpoint to serve the HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Endpoint to get valid options for dropdowns
@app.get("/options")
async def get_options():
    try:
        options = {
            "teams": predictor.label_encoders["team1"].classes_.tolist(),  # Assumes team1 and team2 have same classes
            "venues": predictor.label_encoders["venue"].classes_.tolist()
        }
        # Toss winner uses same options as teams
        options["toss_winner"] = options["teams"]
        return options
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Convert input to dictionary
        input_dict = data.dict()

        # Validate categorical inputs against label encoders
        for col in ['team1', 'team2', 'toss_winner', 'venue']:
            valid_classes = predictor.label_encoders[col].classes_
            if input_dict[col] not in valid_classes:
                raise HTTPException(status_code=400, detail=f"Invalid {col}: {input_dict[col]}. Must be one of {valid_classes.tolist()}")

        # Validate numerical inputs
        if input_dict['inning'] not in [1, 2]:
            raise HTTPException(status_code=400, detail="Inning must be 1 or 2.")
        if input_dict['total_runs'] < 0:
            raise HTTPException(status_code=400, detail="Total runs must be non-negative.")
        if input_dict['is_wicket'] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise HTTPException(status_code=400, detail="Is_wicket must be in the range 0 to 9.")
        if not (0 <= input_dict['over'] <= 20):
            raise HTTPException(status_code=400, detail="Over must be between 0 and 20.")
        if input_dict['team1_games_played'] < 0:
            raise HTTPException(status_code=400, detail="Team 1 games played must be non-negative.")
        if input_dict['team1_games_won'] < 0 or input_dict['team1_games_won'] > input_dict['team1_games_played']:
            raise HTTPException(status_code=400, detail="Team 1 games won must be non-negative and not exceed games played.")
        if input_dict['team2_games_played'] < 0:
            raise HTTPException(status_code=400, detail="Team 2 games played must be non-negative.")
        elif input_dict['team2_games_won'] < 0 or input_dict['team2_games_won'] > input_dict['team2_games_played']:
            raise HTTPException(status_code=400, detail="Team 2 games won must be non-negative and not exceed games played.")

        # Calculate batting and bowling strength
        team1_win_pct = (input_dict['team1_games_won'] / max(input_dict['team1_games_played'], 1)) * 100
        team2_win_pct = (input_dict['team2_games_won'] / max(input_dict['team2_games_played'], 1)) * 100
        batting_strength = (team1_win_pct + team2_win_pct) / 2
        bowling_strength = (team1_win_pct + team2_win_pct) / 2

        # Prepare DataFrame with calculated strengths
        df_dict = {
            'team1': input_dict['team1'],
            'team2': input_dict['team2'],
            'toss_winner': input_dict['toss_winner'],
            'venue': input_dict['venue'],
            'inning': input_dict['inning'],
            'total_runs': input_dict['total_runs'],
            'is_wicket': input_dict['is_wicket'],
            'over': input_dict['over'],
            'batting_strength': batting_strength,
            'bowling_strength': bowling_strength
        }
        df = pd.DataFrame([df_dict])

        # Make prediction
        prediction = predictor.predict(df)
        return {"predicted_target_score": np.floor(float(prediction[0]))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))