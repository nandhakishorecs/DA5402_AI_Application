## IPL Score prediction using Feedforward Neural Networks 

- This web application uses a simple Feedforward Neural Network with ReLU activation to predict IPL score using historic data. The following code snippet defines model in the file in src/model1.py

```python
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
```

- A hyper parameter sweep is done with mlflow and the best model along with the preprcess directives are saved as .pth and .joblib files in the models/ folder <br>
- These files are used in the backend to build the app (check frontend/app.py)