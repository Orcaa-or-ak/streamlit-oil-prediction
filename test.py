import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
# C:/Users/ehero/Desktop/code/ts_js/Project/

# ----------------------------------- Define the Models -----------------------------------

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size * ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ahead):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size * ahead) 
        self.ahead = ahead
        self.output_size = output_size

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.ahead, self.output_size) 
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size * ahead)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the last time step's output for each sequence
        out = out[:, -1, :]
        # Pass through the linear layer and reshape to [batch, ahead, output_size]
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out

# --------------------------------------------------------------------------------------

def predict_XGBoost(X_input_flat, F, target_feature):
    model_file_path = "models/best-XGBRegressor.pkl"

    # ----- 5. Load the XGBoost Model from the .pkl File -----
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    # ----- 6. Predict the Next 7 Days (for all features) -----
    y_pred_flat = model.predict(X_input_flat)

    # Reshape predictions to (ahead, num_features)
    y_pred_norm = y_pred_flat.reshape(ahead, F)

    # ----- 7. Inverse Transform and Extract the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred_norm)
    y_pred_target = y_pred_all_original[:, target_index]

    # ----- 8. Output the Predicted Values for the Target Feature -----
    print("XGB - target feature '{}':".format(target_feature))
    print(y_pred_target)

def predict_RNN(X_input_tensor, target_feature):
    model_file_path = "models/best-RNN.pth"

    # ----- 5. Define the RNN Model and Load the Saved Model -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256             
    num_layers = 3                
    output_size = num_features       

    model = RNN(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=ahead)
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()

    # ----- 6. Predict the Next 7 Days -----
    with torch.no_grad():
        y_pred_tensor = model(X_input_tensor)
    y_pred = y_pred_tensor.squeeze(0).cpu().numpy() 

    # ----- 7. Inverse Transform and Extract the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred)
    y_pred_target = y_pred_all_original[:, target_index]

    # ----- 8. Output the Predicted Values for the Target Feature -----
    print(f"RNN - target feature '{target_feature}':")
    print(y_pred_target)

def predict_LSTM(X_input_tensor, target_feature):
    model_file_path = "models/best-LSTM.pth"
    
    # ----- 5. Load the LSTM Model from the .pth File -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256 
    num_layers = 3 
    output_size = num_features  

    model = LSTM(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=ahead)
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()

    # ----- 6. Predict the Next 7 Days -----
    with torch.no_grad():
        y_pred_tensor = model(X_input_tensor)
    y_pred = y_pred_tensor.squeeze(0).cpu().numpy()

    # ----- 8. Output the Predicted Values for the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred)
    y_pred_target = y_pred_all_original[:, target_index]

    print(f"LSTM - target feature '{target_feature}':")
    print(y_pred_target)

def predict_BiLSTM(X_input_tensor, target_feature):
    model_file_path = "models/best-BiLSTM.pth"
    
    # ----- 5. Load the BiLSTM Model from the .pth File -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256                
    num_layers = 3                
    output_size = num_features       

    model = BiLSTM(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=ahead)
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()

    # ----- 6. Predict the Next 7 Days -----
    with torch.no_grad():
        y_pred_tensor = model(X_input_tensor)
    y_pred = y_pred_tensor.squeeze(0).cpu().numpy()

    # ----- 7. Inverse Transform and Extract the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred)
    y_pred_target = y_pred_all_original[:, target_index]

    # ----- 8. Output the Results -----
    print(f"BiLSTM - target feature '{target_feature}':")
    print(y_pred_target)

def predict_MLP(X_input_tensor, target_feature):
    model_file_path = "models/best-MLP.pth"

    # ----- 5. Define the MLP Model and Load the Saved Model -----
    num_features = len(FEATURES)
    input_size_model = sequence_length * num_features 
    hidden_size = 256  
    output_size = num_features  

    model = MLP(input_size=input_size_model, hidden_size=hidden_size, output_size=output_size, ahead=ahead)
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()

    # ----- 6. Predict the Next 7 Days -----
    with torch.no_grad():
        y_pred_tensor = model(X_input_tensor)
        
    y_pred = y_pred_tensor.squeeze(0).cpu().numpy()

    # ----- 7. Inverse Transform and Extract the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred)
    y_pred_target = y_pred_all_original[:, target_index]

    # ----- 8. Output the Predicted Values for the Target Feature -----
    print(f"MLP - target feature '{target_feature}':")
    print(y_pred_target)

# ----------------------------------------------------------------------    

if __name__ == "__main__":
    train_file_path = "Dataset/UltimateData.csv"
    prediction_file_path = "Dataset/UltimateTest.csv"
    
    sequence_length = 60        # Input: 60 days
    ahead = 7                   # Forecast horizon: next 7 days

    # All features used during training
    FEATURES = ["WTI_Price", "Brent_Price", "DJI", "EUR-USD", "GBP-USD", "CNY-USD", 
                "Gold_Price", "Natural_Gas", "Silver_Price", "SP500", "US10B", "US_Index"]
    
    # Chosen column for predicted
    target_feature = "WTI_Price"

    # ---------------- 1. Fit Scaler on the Training Data ----------------
    train_df = pd.read_csv(train_file_path, parse_dates=["date"])
    train_df.sort_values(by="date", inplace=True)
    train_data = train_df[FEATURES].values

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    target_index = FEATURES.index(target_feature)

    # ---------------- 2. Load Historical Data and Select the Latest 60 Days ----------------
    pred_df = pd.read_csv(prediction_file_path, parse_dates=["date"])

    # Reorder the data based on date in case it's messy
    pred_df.sort_values(by="date", inplace=True)

    # Check if the dataset has at least 60 days of data
    if pred_df.shape[0] < sequence_length:
        raise ValueError(f"Historical data must have at least {sequence_length} days. Found only {pred_df.shape[0]} days.")

    # Select the latest 60 days from the historical data
    latest_60 = pred_df.tail(sequence_length)[FEATURES].values

    # ---------------- 3. Normalize the 60-Day Input ----------------
    latest_60_norm = scaler.transform(latest_60)
    
    # ----- 4. Prepare the Input for the LSTM Model -----
    X_input = np.expand_dims(latest_60_norm, axis=0)  

    # Convert to PyTorch tensor
    X_input_tensor = torch.tensor(X_input, dtype=torch.float32)

    # Flatten the input (for XGB)
    N, L, F = X_input.shape  # Here N=1, L=60, F=num_features
    X_input_flat = X_input.reshape(N, L * F)

    predict_XGBoost(X_input_flat, F, target_feature)
    predict_RNN(X_input_tensor, target_feature)
    predict_LSTM(X_input_tensor, target_feature)
    predict_BiLSTM(X_input_tensor, target_feature)
    predict_MLP(X_input_tensor, target_feature)
