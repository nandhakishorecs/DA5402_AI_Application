# Low-Level Design Document

## 1. Overview
- Facilitate the intricacies of connections between different modules in the application

## 2. Component-Level Design
### 2.1 Frontend (`frontend`)
- UI Breakdown: 
    - Background Image: uses a standard .jpeg background image 
    - Form: Uses a two coloumn form with gaurd rails to get proper input from the user 
    - About Button: Description about the app 

### 2.2 Backend (`src`)
- APIs: 
    - handles the inputs and gets predictions using RESTAPI 
    - GET /
        - Serves the index.html UI from frontend/.

    - GET /options
        - Returns valid dropdown values (teams, venues, toss winners).

    - POST /predict
        - Performs input validation and returns target score prediction.

    - GET /metrics
        - Exposes Prometheus metrics for monitoring.

- CricketTargetScorePredictor (`src/model.py`)
    - Loads trained model (.pth)

- Loads preprocessing pipeline (.joblib)
    - Handles encoding/decoding

- Exposes predict() method
    - FastAPI App (src/app.py)

- Mounts static frontend
    - Handles CORS

- Integrates Prometheus middleware
    - Serves and validates prediction requests

### 2.3 DAGs (`dags`)
- DAG definition 
    - Ingest data from the .csv files 
    - Check for noisy data and make necessary changes to merge the necessary column names. 
    - merge Ball by Ball data and meta data 
    - Handle drift , if needed. 

### 2.4 Data Analysis (`data_analysis`)
- Uses a .ipynb script to check the quality of data. 

### 2.5 Models (`models`)
- Stores the best version of the model as .pth file 
- Stores the aiding pacages (label encoders, sclaers) as a .joblib file 

### 2.6 Monitoring
- Uses prometheus based monitering to expose the metrics: 
    - Memory Usage 
    - Number of HTTP requests 
- A Grafana based dashboard is implemented to monitor the app 

### 2.7 Tests
- Testing framework (e.g., Pytest, Unittest)
- Test coverage strategy
- Mocking/stubbing external services

## 3. Code Quality and Conventions
- Naming conventions
    - The UI handles team names and venues without repetitions
- The modules of the application are stored in separate directories for ease of handling 

## 4. Change Management
- Versioning approach 
    - Uses Git to tag the models and datasets 