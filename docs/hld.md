# High-Level Design Document

## 1. Overview
- Purpose: A web application to predict score of an IPL match 

## 2. Architecture Overview
- Diagram of the overall architecture
- Tech stack 
    - Frontend: HTML, CSS, Javascript 
    - Backend: FASTAPI, Joblib
    - Model: PyTorch, MLFLOW

## 3. Module Breakdown
### 3.1 Frontend
- Provides Basic UI 

### 3.2 Backend (`src`)
- Application framework using FASTAPI 

### 3.3 Data Analysis (`data_analysis`)
- Purpose (e.g., EDA, reports, dashboards)
- Tools/libraries used: Pandas, Numpy, airflow 

### 3.4 Models
- Uses Artificial Neural Network 
- Integration with backend

### 3.5 Monitoring
- Metrics tracked (API usage, HTTP calls, useage of memory)
- Tools used: Prometheus, Grafana

### 3.6 DAGs
- Utilises Airflow to process the dataset and clear data 

## 4. Deployment Strategy
- Deployed as a Docker image 