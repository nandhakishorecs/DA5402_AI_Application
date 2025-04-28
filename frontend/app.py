from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from starlette.responses import Response
import time
import random

# Import the CricketTargetScorePredictor class from src/model.py
from model1 import CricketTargetScorePredictor

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, "..", "frontend")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Latency',
    ['endpoint']
)
PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Prediction Endpoint Latency',
    ['endpoint']
)
PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Total Prediction Errors',
    ['endpoint']
)

# Middleware to track request count and latency for all endpoints
@app.middleware("http")
async def add_prometheus_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

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
MODEL_PATH = os.path.join("..", "models", "target_score_model_best.pth")
PREPROCESS_PATH = os.path.join("..", "models", "preprocess_best.joblib")

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
        
        # Validate over: must be between 0 and 20, with valid decimal increments (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        over = input_dict['over']
        if not (0 <= over <= 20):
            raise HTTPException(status_code=400, detail="Over must be between 0 and 20.")
        integer_part = int(over)
        decimal_part = round(over - integer_part, 1)
        if decimal_part not in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            raise HTTPException(status_code=400, detail="Over decimal part must be one of [0.0, 0.1, 0.2, 0.3, 0.4, 0.5].")

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

        # Make prediction with latency tracking
        start_time = time.time()
        try:
            prediction = predictor.predict(df)
            PREDICTION_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
            pred = np.abs(np.floor(float(prediction[0])))
            if(pred>= 250): 
                pred = random.uniform(19, 250)
            elif(pred < 250 and pred > 110): 
                pred = random.uniform(110, 249)
            elif(pred < 110 and pred > 40): 
                pred = random.uniform(40, 109)
            return {"predicted_target_score": np.abs(pred)}
        except Exception as e:
            PREDICTION_ERRORS.labels(endpoint="/predict").inc()
            raise

    except Exception as e:
        PREDICTION_ERRORS.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}