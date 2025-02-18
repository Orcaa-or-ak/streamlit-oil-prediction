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
def predict_xgboost(df_filtered, features):
    model = xgb.Booster()
    model.load_model("model/xgboost_model.json")
    latest_input = df_filtered[features].iloc[-1].to_numpy().reshape(1, -1)
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
    df_numeric = df.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce").dropna()
    
    if len(df_numeric) < SEQ_LENGTH:
        st.error("⚠️ Not enough data in the selected date range for LSTM prediction.")
        return None

    input_seq = df_numeric.iloc[-SEQ_LENGTH:].to_numpy(dtype=np.float32)
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

st.markdown("<p class='big-font'><b>Oil Price Prediction</b></p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Ensure the CSV file contains columns: date, WTI_Price, Brent_Price, USDPrice")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df = df.sort_values(by="date")

    df["date"] = pd.to_datetime(df["date"])
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    # Mặc định chọn 30 ngày cuối cùng
    default_start = max(min_date, max_date - pd.Timedelta(days=30))

    st.markdown("Choose a date range in **YYYY/MM/DD** format.")

    # Chọn ngày bắt đầu và kết thúc
    start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date, format="YYYY/MM/DD")
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, format="YYYY/MM/DD")

    # Giới hạn không quá 30 ngày
    if (end_date - start_date).days > 30:
        st.warning("⚠️ You can select a maximum of 30 days. Adjusting your selection.")
        end_date = start_date + pd.Timedelta(days=30)

    df_filtered = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    df_filtered["WTI_EMA"] = calculate_ema(df_filtered["WTI_Price"], span=10)
    df_filtered["USD_EMA"] = calculate_ema(df_filtered["USDPrice"], span=10)
    FEATURES = ["WTI_Price", "Brent_Price", "USDPrice", "WTI_EMA", "USD_EMA"]
    df_display = df_filtered[["date"] + FEATURES]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Data Preview")
        st.dataframe(df_display.tail(10))
    
    with col2:
        model_choice = st.radio("Select Model", ("XGBoost", "LSTM"))
        if st.button("Predict"):
            if len(df_filtered) > 0:
                if model_choice == "XGBoost":
                    prediction = predict_xgboost(df_filtered, FEATURES)
                else:
                    prediction = predict_bilstm(df_filtered)
                
                if prediction is not None:
                    st.success(f"Predicted Next Day Price ({model_choice}): {prediction:.4f}")
            else:
                st.error("No data available in the selected range. Choose a different date range.")
    
    # Trend visualization
    st.write("### Brent Price Trend")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x=df_filtered["date"], y=df_filtered["Brent_Price"], marker='o', linestyle='-', color='blue', ax=ax)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Brent Price", fontsize=12)
    ax.set_title("Brent Price Trend in Selected Range", fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write("### Data Used for Prediction")
    st.dataframe(df_display.tail(30))
