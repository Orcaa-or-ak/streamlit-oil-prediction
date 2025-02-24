import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# C:/Users/ehero/Desktop/code/ts_js/Project/

torch.classes.__path__ = []

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

def predict_XGBoost(X_input_flat, F):
    model_file_path = "models/best-XGBRegressor.pkl"

    # ----- 5. Load the XGBoost Model from the .pkl File -----
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    # ----- 6. Predict the Next 7 Days (for all features) -----
    y_pred_flat = model.predict(X_input_flat)

    # Reshape predictions to (ahead, num_features)
    y_pred_norm = y_pred_flat.reshape(7, F)

    # ----- 7. Inverse Transform and Extract the Target Feature -----
    y_pred_all_original = scaler.inverse_transform(y_pred_norm)
    y_pred_target = y_pred_all_original[:, target_index]

    # ----- 8. Output the Predicted Values for the Target Feature -----
    return y_pred_target

def predict_RNN(X_input_tensor):
    model_file_path = "models/best-RNN.pth"

    # ----- 5. Define the RNN Model and Load the Saved Model -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256             
    num_layers = 3                
    output_size = num_features       

    model = RNN(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=7)
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
    return y_pred_target

def predict_LSTM(X_input_tensor):
    model_file_path = "models/best-LSTM.pth"
    
    # ----- 5. Load the LSTM Model from the .pth File -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256 
    num_layers = 3 
    output_size = num_features  

    model = LSTM(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=7)
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
    return y_pred_target

def predict_BiLSTM(X_input_tensor):
    model_file_path = "models/best-BiLSTM.pth"
    
    # ----- 5. Load the BiLSTM Model from the .pth File -----
    num_features = len(FEATURES)
    input_size_model = num_features  
    hidden_size = 256                
    num_layers = 3                
    output_size = num_features       

    model = BiLSTM(input_size=input_size_model, hidden_size=hidden_size,
                output_size=output_size, num_layers=num_layers, ahead=7)
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
    return y_pred_target

def predict_MLP(X_input_tensor):
    model_file_path = "models/best-MLP.pth"

    # ----- 5. Define the MLP Model and Load the Saved Model -----
    num_features = len(FEATURES)
    input_size_model = sequence_length * num_features 
    hidden_size = 256  
    output_size = num_features  

    model = MLP(input_size=input_size_model, hidden_size=hidden_size, output_size=output_size, ahead=7)
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
    return y_pred_target

# ----------------------------------------------------------------------    

# Streamlit UI
st.set_page_config(page_title="Oil Price Prediction", layout="wide")
st.title("Oil Price Prediction")

# Create layout with two columns
col1, col2 = st.columns([1, 3])

with col1:
    # Sidebar Input Section
    st.header("Input Section")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        pred_df = pd.read_csv(uploaded_file, parse_dates=["date"])
        pred_df.sort_values(by="date", inplace=True)
        pred_df["date"] = pred_df["date"].dt.date  # Ensure only date without timestamp
        
        # Dropdown for model selection
        model_choice = st.selectbox("Choose a model", ["XGBoost", "RNN", "LSTM", "BiLSTM", "MLP"])
        
        # Dropdown for feature selection
        FEATURES = ["WTI_Price", "Brent_Price", "DJI", "EUR-USD", "GBP-USD", "CNY-USD", 
                    "Gold_Price", "Natural_Gas", "Silver_Price", "SP500", "US10B", "US_Index"]
        target_feature = st.selectbox("Select feature to predict", FEATURES)
        
        # Load training data for scaler fitting
        train_file_path = "Dataset/UltimateData.csv"
        train_df = pd.read_csv(train_file_path, parse_dates=["date"])
        train_df.sort_values(by="date", inplace=True)
        train_data = train_df[FEATURES].values
        
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        target_index = FEATURES.index(target_feature)
        
        # Select date range for prediction
        st.subheader("Choose a date range in YYYY/MM/DD format.")
        start_date = st.date_input("Start Date", min_value=pred_df["date"].min(), max_value=pred_df["date"].max())
        end_date = st.date_input("End Date", min_value=pred_df["date"].min(), max_value=pred_df["date"].max())
        
        if (end_date - start_date).days < 60:
            st.error("The selected date range must be at least 60 days. Please choose again.")
        else:
            filtered_data = pred_df[(pred_df["date"] >= start_date) & (pred_df["date"] <= end_date)]
        
            # Select number of days to display
            display_days = st.slider("Select number of days to display", 1, 7, 7)
            
            # Data preprocessing
            sequence_length = 60
            ahead = 7
            
            latest_60 = filtered_data.tail(sequence_length)[FEATURES].values
            latest_60_norm = scaler.transform(latest_60)
            X_input = np.expand_dims(latest_60_norm, axis=0)
            X_input_tensor = torch.tensor(X_input, dtype=torch.float32)
            X_input_flat = X_input.reshape(1, -1)
            
            # Placeholder for results in col2
            with col2:
                st.header("Prediction Results")
                table_placeholder = st.empty()
                chart_placeholder = st.empty()
                data_table_placeholder = st.empty()
            
            # Prediction button
            if st.button("Predict"):
                # Placeholder for model prediction logic
                if model_choice == "RNN":
                    predictions = predict_RNN(X_input_tensor)
                elif model_choice == "LSTM":
                    predictions = predict_LSTM(X_input_tensor)
                elif model_choice == "BiLSTM":
                    predictions = predict_BiLSTM(X_input_tensor)
                elif model_choice == "MLP":
                    predictions = predict_MLP(X_input_tensor)
                else:
                    predictions = predict_XGBoost(X_input_flat, len(FEATURES))
                
                last_date = pd.to_datetime(filtered_data["date"].iloc[-1])
                future_dates = pd.date_range(start=last_date, periods=ahead+1)[1:]
                
                # Extend predictions with last actual data point for continuity
                extended_dates = pd.to_datetime(np.concatenate([[last_date], future_dates[:display_days]]))
                extended_predictions = np.concatenate([[filtered_data[target_feature].iloc[-1]], predictions[:display_days]])
                
                with col2:
                    prediction_table = pd.DataFrame({"Date": future_dates[:display_days], target_feature: predictions[:display_days]})
                    prediction_table["Date"] = prediction_table["Date"].dt.strftime("%Y-%m-%d")
                    table_placeholder.dataframe(prediction_table.set_index("Date"))
                    
                    # Plot feature trend
                    fig, ax = plt.subplots(figsize=(16, 8))
                    ax.plot(pd.to_datetime(filtered_data['date'].tail(sequence_length)), filtered_data[target_feature].tail(sequence_length), label="Actual")
                    ax.plot(extended_dates, extended_predictions, label="Predicted", linestyle="dashed", marker='o')
                    ax.set_title(f"{target_feature} Trend")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(target_feature)
                    ax.legend()
                    chart_placeholder.pyplot(fig)
                    
                    # Show data used for prediction
                    data_table_placeholder.dataframe(filtered_data.set_index("date"))
