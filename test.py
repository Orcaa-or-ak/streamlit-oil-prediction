import xgboost as xgb
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

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

def calculate_ema(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

# Function for XGBoost prediction
def predict_xgboost(latest_input, features):
    model = xgb.Booster()
    model.load_model("xgboost_model.json")

    dinput = xgb.DMatrix(latest_input, feature_names=features)
    prediction = model.predict(dinput)

    print(f"Predicted Next Day Price (XGBoost): {prediction[0]:.4f}")

# Function for BiLSTM prediction
def predict_bilstm(df):
    input_size = 5
    hidden_size = 256
    num_layers = 3
    output_size = 1
    ahead = 1
    model_path = "best-BiLSTM.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTM(input_size, hidden_size, output_size, num_layers, ahead)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    SEQ_LENGTH = 30
    input_seq = df[-SEQ_LENGTH:].to_numpy()

    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()

    print(f"Predicted Next Day Price (BiLSTM): {prediction[0][0][0]:.4f}")

# Main execution
if __name__ == "__main__":
    file_path = "Book1.csv"
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values(by="date")

    df["WTI_EMA"] = calculate_ema(df["WTI_Price"], span=10)
    df["USD_EMA"] = calculate_ema(df["USDPrice"], span=10)

    FEATURES = ["WTI_Price", "Brent_Price", "USDPrice", "WTI_EMA", "USD_EMA"]
    df = df[FEATURES].dropna()  # Loại bỏ dòng có giá trị NaN

    latest_input = df.iloc[-1].to_numpy().reshape(1, -1)

    predict_xgboost(latest_input, FEATURES)
    predict_bilstm(df)
