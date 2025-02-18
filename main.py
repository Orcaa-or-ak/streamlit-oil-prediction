import streamlit as st
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.classes.__path__ = []

# Define BiLSTM model
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

# Function to calculate EMA
def calculate_ema(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

# Function for XGBoost prediction
def predict_xgboost(latest_input, features):
    model = xgb.Booster()
    model.load_model("model/xgboost_model.json")
    dinput = xgb.DMatrix(latest_input, feature_names=features)
    prediction = model.predict(dinput)
    return prediction[0]

# Function for BiLSTM prediction
def predict_bilstm(df):
    input_size = 5
    hidden_size = 256
    num_layers = 3
    output_size = 1
    ahead = 1
    model_path = "model/best-BiLSTM.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiLSTM(input_size, hidden_size, output_size, num_layers, ahead)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    SEQ_LENGTH = 30
    input_seq = df.iloc[-SEQ_LENGTH:].to_numpy()
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    
    return prediction[0][0][0]

# Streamlit UI
st.set_page_config(page_title="Oil Price Prediction", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size:30px !important; }
    .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<p class='big-font'><b>Oil Price Prediction Web App</b></p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Ensure the CSV file contains columns: date, WTI_Price, Brent_Price, USDPrice")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df = df.sort_values(by="date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df["WTI_EMA"] = calculate_ema(df["WTI_Price"], span=10)
    df["USD_EMA"] = calculate_ema(df["USDPrice"], span=10)
    FEATURES = ["WTI_Price", "Brent_Price", "USDPrice", "WTI_EMA", "USD_EMA"]
    df_display = df[["date"] + FEATURES]
    df_model = df[FEATURES].dropna()
    latest_input = df_model.iloc[-1].to_numpy().reshape(1, -1)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Data Preview")
        st.dataframe(df_display.tail(10))
    
    with col2:
        model_choice = st.radio("Select Model", ("XGBoost", "BiLSTM"))
        if st.button("Predict"):
            prediction = predict_xgboost(latest_input, FEATURES) if model_choice == "XGBoost" else predict_bilstm(df_model)
            st.success(f"Predicted Next Day Price ({model_choice}): {prediction:.4f}")
    
    # Trend visualization
    st.write("### Brent Price Trend")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=df["date"].iloc[-30:], y=df["Brent_Price"].iloc[-30:], marker='o', linestyle='-', color='blue', ax=ax)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Brent Price", fontsize=12)
    ax.set_title("Brent Price Trend (Last 30 Days)", fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write("### Data Used for Prediction")
    st.dataframe(df_display.tail(30))
