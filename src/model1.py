import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import mlflow
import mlflow.pytorch
from itertools import product

class CricketTargetScorePredictor:
    def __init__(self, hidden_dims=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001):
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.categorical_cols = ['team1', 'team2', 'toss_winner', 'venue']
        self.numerical_cols = ['inning', 'total_runs', 'is_wicket', 'over', 
                             'batting_strength', 'bowling_strength']

    def _build_model(self, input_dim):
        layers = []
        in_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        return nn.Sequential(*layers).to(self.device)

    def _preprocess_data(self, df, fit=False):
        X = df.copy()
        y = None
        
        # Drop match_id and target_score if they exist
        columns_to_drop = [col for col in ['match_id', 'target_score'] if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            if 'target_score' in columns_to_drop:
                y = df['target_score'].values
        
        # Encode categorical variables
        if fit:
            for col in self.categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
        else:
            for col in self.categorical_cols:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Standardize numerical features
        if fit:
            X[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        else:
            X[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        
        return X, y

    def fit(self, df, epochs=10, batch_size=32, validation_split=0.2, 
            model_file_path='../models/target_score_model.pth', 
            preprocess_file_path='../models/preprocess.joblib'):
        X, y = self._preprocess_data(df, fit=True)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = self._build_model(input_dim=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss = train_loss / len(train_dataset)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model_file_path, preprocess_file_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
        
        return best_val_loss

    def save_model(self, model_file_path='../models/target_score_model.pth', 
                   preprocess_file_path='../models/preprocess.joblib'):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        
        torch.save(self.model.state_dict(), model_file_path)
        mlflow.log_artifact(model_file_path)
        print(f"Model state saved to {model_file_path}")
        
        preprocess_objects = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'hidden_dims': self.hidden_dims
        }
        joblib.dump(preprocess_objects, preprocess_file_path)
        mlflow.log_artifact(preprocess_file_path)
        print(f"Preprocessing objects saved to {preprocess_file_path}")

    def load_model(self, model_file_path='../models/target_score_model.pth', 
                   preprocess_file_path='../models/preprocess.joblib'):
        preprocess_objects = joblib.load(preprocess_file_path)
        self.label_encoders = preprocess_objects['label_encoders']
        self.scaler = preprocess_objects['scaler']
        self.hidden_dims = preprocess_objects['hidden_dims']
        print(f"Preprocessing objects loaded from {preprocess_file_path}")
        
        # Input dimension is the sum of categorical and numerical columns
        input_dim = len(self.categorical_cols) + len(self.numerical_cols)
        self.model = self._build_model(input_dim=input_dim)
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model state loaded from {model_file_path}")

    def predict(self, df):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        
        self.model.eval()
        X, _ = self._preprocess_data(df, fit=False)
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions

def run_sweep(df, hyperparams):
    mlflow.set_experiment("CricketTargetScorePrediction")
    
    best_val_loss = float('inf')
    best_run_id = None
    best_model_path = 'models/target_score_model_best.pth'
    best_preprocess_path = 'models/preprocess_best.joblib'
    
    param_combinations = list(product(
        hyperparams['hidden_dims'],
        hyperparams['dropout_rate'],
        hyperparams['learning_rate']
    ))
    
    for hidden_dims, dropout_rate, learning_rate in param_combinations:
        with mlflow.start_run():
            mlflow.log_param("hidden_dims", hidden_dims)
            mlflow.log_param("dropout_rate", dropout_rate)
            mlflow.log_param("learning_rate", learning_rate)
            
            predictor = CricketTargetScorePredictor(
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            model_file_path = f'../models/target_score_model_{mlflow.active_run().info.run_id}.pth'
            preprocess_file_path = f'../models/preprocess_{mlflow.active_run().info.run_id}.joblib'
            
            val_loss = predictor.fit(
                df,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                model_file_path=model_file_path,
                preprocess_file_path=preprocess_file_path
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_run_id = mlflow.active_run().info.run_id
                os.rename(model_file_path, best_model_path)
                os.rename(preprocess_file_path, best_preprocess_path)
                mlflow.log_metric("best_val_loss", best_val_loss)
    
    print(f"Best run ID: {best_run_id}, Best validation loss: {best_val_loss}")
    return best_model_path, best_preprocess_path, best_run_id

# Example usage
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    df = pd.read_csv('/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA5402/DA5402_AI_Application/dags/ipl_data_processed.csv')
    # Define hyperparameter grid for sweep
    hyperparams = {
        'hidden_dims': [[128, 64, 32], [256, 128, 64]],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.0005]
    }
    
    # Run MLflow sweep
    best_model_path, best_preprocess_path, best_run_id = run_sweep(df, hyperparams)
    
    # Load the best model for predictions
    predictor = CricketTargetScorePredictor()
    predictor.load_model(best_model_path, best_preprocess_path)
    
    # Make predictions with the best model
    test_df = df.drop(['target_score', 'match_id'], axis=1).iloc[:5]
    predictions = predictor.predict(test_df)
    print("\nSample Predictions (Best Model):", predictions)