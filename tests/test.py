import pytest
from fastapi.testclient import TestClient

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))

import warnings
warnings.filterwarnings('ignore')

from app import app

client = TestClient(app)

def test_predict_success():
    valid_input = {
        "team1": "Chennai Super Kings",
        "team2": "Mumbai Indians",
        "toss_winner": "Chennai Super Kings",
        "venue": "SuperSport Park, South Africa",
        "inning": 1,
        "total_runs": 85,
        "is_wicket": 3,
        "over": 9.3,
        "team1_games_played": 14,
        "team1_games_won": 10,
        "team2_games_played": 14,
        "team2_games_won": 8
    }

    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert "predicted_target_score" in response.json()
    assert isinstance(response.json()["predicted_target_score"], (int, float))
