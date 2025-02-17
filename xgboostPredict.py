import xgboost as xgb
import pandas as pd

model = xgb.Booster()
model.load_model("xgboost_model.json")  

file_path = "Book1.csv"  
df = pd.read_csv(file_path, parse_dates=["date"])
df = df.sort_values(by="date")

def calculate_ema(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

df["WTI_EMA"] = calculate_ema(df["WTI_Price"], span=10)  
df["USD_EMA"] = calculate_ema(df["USDPrice"], span=10)  

FEATURES = ["WTI_Price", "Brent_Price", "USDPrice", "WTI_EMA", "USD_EMA"]  

df = df[FEATURES] 

latest_input = df.iloc[-1].to_numpy().reshape(1, -1)  

dinput = xgb.DMatrix(latest_input, feature_names=FEATURES)

prediction = model.predict(dinput)

print(f"Predicted Next Day Price: {prediction[0]:.4f}")
