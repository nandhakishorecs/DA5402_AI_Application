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
BALL_BY_BALL_FILE = os.path.join(DAGS_DIR, 'ipl_deliveries.csv')
MATCHES_FILE = os.path.join(DAGS_DIR, 'ipl_matches.csv')
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
        ball_by_ball.to_csv(os.path.join(OUTPUT_DIR, 'deliveries.csv'), index=False)
        matches.to_csv(os.path.join(OUTPUT_DIR, 'matches.csv'), index=False)
        logger.info("Data ingested successfully.")
    except Exception as e:
        logger.error(f"Error ingesting data: {str(e)}")
        raise

def replace_elements(df, replacement_dict, col_names, filename, output_path=None):
    """
    Replace values in 'team1' and 'team2' columns based on a user-provided dictionary.

    Args:
        csv_path (str): Path to the input CSV file.
        replacement_dict (dict): Dictionary where keys are old names and values are new names.
        output_path (str, optional): Path to save the updated CSV. If None, overwrites the input CSV.
    """
    # Replace in both columns
    for column in col_names: 
        if column in df.columns:
            df[column] = df[column].map(replacement_dict).fillna(df[column])
        else:
            raise ValueError(f"Column '{column}' not found in CSV.")
    
    # Save the updated CSV
    if output_path is None:
        output_path = OUTPUT_DIR  # overwrite
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    
    print(f"Replacement done! File saved at: {output_path}")

def check_duplicates(): 
    try:
        logger.info("Checking for duplicates in dataset ...") 
        ball_by_ball = pd.read_csv(os.path.join(OUTPUT_DIR, 'deliveries.csv'))
        matches = pd.read_csv(os.path.join(OUTPUT_DIR, 'matches.csv'))

        team_replacement = {
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
            'Kings XI Punjab': 'Punjab Kings',
            'Deccan Chargers': 'Sunrisers Hyderabad', 
            'Delhi Daredevils' : 'Delhi Capitals',
            'Gujarat Titans' : 'Gujarat Lions'
        }

        venue_replacement = {
            'Arun Jaitley Stadium': 'Arun Jaitley Stadium, Delhi',
            'Barabati Stadium': 'Barabati Stadium, Odissa',
            'Brabourne Stadium': 'Brabourne Stadium, Mumbai', 
            'Buffalo Park': 'Buffalo Park, South Africa',
            'De Beers Diamond Oval': 'De Beers Diamond Oval, South Africa', 
            'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy, Mumbai',
            'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
            'Eden Gardens': 'Eden Gardens, Kolkata', 
            'Feroz Shah Kotla': 'Feroz Shah Kotla, Delhi',
            'Green Park': 'Green Park, Kanpur',
            'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium, Dharamsala',
            'Holkar Cricket Stadium': 'Holkar Cricket Stadium, Indore', 
            'JSCA International Stadium Complex' : 'JSCA International Stadium Complex, Ranchi',
            'Kingsmead': 'Kingsmead, South Africa', 
            'M Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
            'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
            'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk, Chennai', 
            'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chepauk, Chennai',
            'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium, Pune',
            'Nehru Stadium': 'Nehru Stadium, Guwahati',
            'New Wanderers Stadium': 'New Wanderers Stadium, South Africa', 
            'Newlands': 'Newlands, South Africa', 
            'OUTsurance Oval': 'OUTsurance Oval, South Africa',
            'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
            'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
            'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
            'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
            'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
            'Saurashtra Cricket Association Stadium': 'Saurashtra Cricket Association Stadium, Rajkot', 
            'Sawai Mansingh Stadium':'Sawai Mansingh Stadium, Jaipur',
            'Shaheed Veer Narayan Singh International Stadium': 'Shaheed Veer Narayan Singh International Stadium, New Raipur',
            'Sheikh Zayed Stadium': 'Zayed Cricket Stadium, Abu Dhabi',
            "St George's Park": "St George's Park, South Africa", 
            'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium, Pune',
            'SuperSport Park':'SuperSport Park, South Africa', 
            'Wankhede Stadium': 'Wankhede Stadium, Mumbai',
        }

        replace_elements(ball_by_ball, team_replacement, ['team1', 'team2'] , filename = 'deliveries.csv')
        replace_elements(matches, team_replacement, ['team1', 'team2', 'toss_winner'], filename = 'matches.csv')
        replace_elements(matches, venue_replacement, ['venue'], filename = 'matches.csv')

    except Exception as e:
        logger.error(f"Error removing suplicate data in dataset: {str(e)}")
        raise

def drop_elements_in_column(df, columns, elements_to_drop, filename, output_path=None):

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: np.nan if x == elements_to_drop else x
            )
        else:
            print(f"Warning: Column '{col}' not found in dataframe.")

    # Save the updated CSV
    if output_path is None:
        output_path = OUTPUT_DIR  # overwrite
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    
    logger.info(f"Dropped elements {elements_to_drop} from column '{columns}'. File saved at: {output_path}")

def drop_data(): 
    try:
        logger.info("Checking for old data and dropping unwanted data in dataset ...") 
        ball_by_ball = pd.read_csv(os.path.join(OUTPUT_DIR, 'deliveries.csv'))
        matches = pd.read_csv(os.path.join(OUTPUT_DIR, 'matches.csv'))

        elements = [
            'Pune Warriors', 
            'Rising Pune Supergiants', 
            'Rising Pune Supergiant', 
            'Kochi Tuskers Kerala'
        ]

        drop_elements_in_column(matches, ['team1', 'team2', 'toss_winner'], elements, filename = 'matches.csv')
        drop_elements_in_column(ball_by_ball, ['team1', 'team2'], elements, filename = 'deliveries.csv')

    except Exception as e:
        logger.error(f"Error dropping data in dataset: {str(e)}")
        raise

def preprocess_data():
    """Preprocess IPL dataset to create features for model training."""
    try:
        logger.info("Preprocessing IPL dataset...")
        ball_by_ball = pd.read_csv(os.path.join(OUTPUT_DIR, 'deliveries.csv'))
        matches = pd.read_csv(os.path.join(OUTPUT_DIR, 'matches.csv'))

        # Aggregate ball-by-ball data per match and inning
        match_features = ball_by_ball.groupby(['match_id', 'inning']).agg({
            'total_runs': 'sum',
            'is_wicket': 'max',
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
        # features = ['total_runs', 'is_wicket', 'over', 'batting_strength', 'bowling_strength']
        # match_features[features] = (match_features[features] - match_features[features].mean()) / match_features[features].std()

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
    'final_data_pipeline',
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
    clean_data_task = PythonOperator(
        task_id='check_duplicates',
        python_callable = check_duplicates,
    )
    drop_data_task = PythonOperator(
        task_id='remove_old_data', 
        python_callable = drop_data
    )
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    ingest_task >> clean_data_task >> drop_data_task >> preprocess_task