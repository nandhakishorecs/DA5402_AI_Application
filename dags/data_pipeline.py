from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DAGS_DIR = '/opt/airflow/dags'  # Airflow DAGs folder where CSV files are stored
BALL_BY_BALL_FILE = os.path.join(DAGS_DIR, 'deliveries.csv')
MATCHES_FILE = os.path.join(DAGS_DIR, 'matches.csv')
OUTPUT_DIR = '/opt/airflow/dags'

def check_file_exists(file_path):
    """Check if a file exists at the given path."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

def ingest_data():
    """Ingest IPL dataset from CSV files in the DAGs folder."""
    try:
        logger.info("Ingesting IPL dataset...")
        # Check if files exist
        check_file_exists(BALL_BY_BALL_FILE)
        check_file_exists(MATCHES_FILE)

        # Read dataset
        ball_by_ball = pd.read_csv(BALL_BY_BALL_FILE)
        matches = pd.read_csv(MATCHES_FILE)

        # Save to temporary location for downstream tasks
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ball_by_ball.to_csv(os.path.join(OUTPUT_DIR, 'ipl_ball_by_ball.csv'), index=False)
        matches.to_csv(os.path.join(OUTPUT_DIR, 'ipl_matches.csv'), index=False)
        logger.info("Data ingested successfully.")
    except Exception as e:
        logger.error(f"Error ingesting data: {str(e)}")
        raise

def preprocess_data():
    """Preprocess IPL dataset to create features for model training."""
    try:
        logger.info("Preprocessing IPL dataset...")
        ball_by_ball = pd.read_csv(os.path.join(OUTPUT_DIR, 'ipl_ball_by_ball.csv'))
        matches = pd.read_csv(os.path.join(OUTPUT_DIR, 'ipl_matches.csv'))

        # Aggregate ball-by-ball data per match and inning
        match_features = ball_by_ball.groupby(['match_id', 'inning']).agg({
            'total_runs': 'sum',
            'is_wicket': 'sum',
            'over': 'max',
        }).reset_index()

        # Filter for first innings only (to predict final score)
        match_features = match_features[match_features['inning'] == 1]

        # Merge with match data to get team info
        match_features = match_features.merge(
            matches[['match_id', 'team1', 'team2', 'toss_winner', 'venue']],
            on='match_id',
            how='left'
        )

        # Calculate team strength (simplified: based on historical win percentage)
        team_wins = matches['winner'].value_counts()
        total_matches = matches.groupby('team1').size() + matches.groupby('team2').size()
        team_strength = (team_wins / total_matches).fillna(0.5)  # Default strength 0.5 if no data

        # Add batting and bowling team strengths
        match_features['batting_strength'] = match_features['team1'].map(team_strength)
        match_features['bowling_strength'] = match_features['team2'].map(team_strength)

        # Handle missing values
        match_features.fillna({'batting_strength': 0.5, 'bowling_strength': 0.5}, inplace=True)

        # Normalize features
        features = ['total_runs', 'is_wicket', 'over', 'batting_strength', 'bowling_strength']
        match_features[features] = (match_features[features] - match_features[features].mean()) / match_features[features].std()

        # Target: Total runs scored in the first inning
        match_features['target_score'] = match_features['total_runs']

        # Save preprocessed data
        match_features.to_csv(os.path.join(OUTPUT_DIR, 'ipl_data_processed.csv'), index=False)
        logger.info("Data preprocessed successfully.")
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

# Define the DAG
with DAG(
    'ipl_data_pipeline_v4',
    start_date=datetime(2025, 4, 26),
    schedule_interval='@daily',
    catchup=False,
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
    }
) as dag:
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
    )
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    ingest_task >> preprocess_task