#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from binance.client import Client
from sklearn.metrics import accuracy_score
from datetime import datetime

########################
# 1. Load .env, Setup Binance Client
########################

load_dotenv()  # Loads credentials from .env

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Create the client instance
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)


########################
# 2. Define Your Model
########################

class PricePredictor(nn.Module):
    VERSION = 3
    
    def __init__(self, input_size, is_stablecoin=False):
        super().__init__()
        
        if is_stablecoin:
            # Example stablecoin architecture
            self.network = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            # Example architecture for non-stablecoins
            self.network = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(256),
                
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(128),
                
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.BatchNorm1d(64),
                
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32),
                
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        return self.network(x)

########################
# 3. Feature Engineering
########################

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal

def calculate_bollinger_position(prices, window=20):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    position = (prices - lower_band) / (upper_band - lower_band)
    return position.clip(0, 1)

def compute_features(df):
    """
    Reproduce the same transformations used during training.
    """
    # Market Structure
    df['depth_imbalance'] = (
        (df['bid_depth'] - df['ask_depth']) /
        (df['bid_depth'] + df['ask_depth'])
    ).clip(-0.5, 0.5)
    
    # Convert total liquidity to log scale
    df['total_liquidity'] = np.log1p(df['total_liquidity'])
    
    # Market Dynamics
    df['returns'] = df['close'].pct_change().clip(-0.05, 0.05)
    df['volatility_15m'] = df['returns'].rolling(4).std()
    df['volume_momentum'] = (
        df['volume'] / df['volume'].rolling(96).mean()
    ).clip(0.1, 10)
    
    # Price Action
    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    df['vwap_deviation'] = ((df['close'] - df['vwap']) / df['vwap']).clip(-0.05, 0.05)
    df['price_momentum_15m'] = df['returns'].rolling(4).mean()
    
    # Technical Indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], _ = calculate_macd(df['close'])
    df['bollinger_position'] = calculate_bollinger_position(df['close'])
    
    # Target
    df['future_return'] = df['close'].shift(-16) / df['close'] - 1
    df['entry_success'] = (df['future_return'] > 0.015).astype(int)
    
    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df


########################
# 4. Fetch Data (2022-2024)
########################

def fetch_2022_to_2024(symbol="BATUSDT"):
    """
    Fetch 15m klines from Binance for the period 2022-01-01 to 2024-01-01.
    This will include all of 2022 and 2023.
    """
    start_str = "2022-01-01"
    end_str   = "2024-01-01"  # up to but not including Jan 1, 2024
    
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_15MINUTE,
        start_str=start_str,
        end_str=end_str
    )
    
    if not klines:
        raise ValueError(f"No klines returned for {symbol} in [2022-2024).")
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base_volume", "taker_quote_volume", "ignore"
    ])
    
    # Convert numeric cols
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_base_volume", "taker_quote_volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    # Convert timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    
    # Approximate or placeholder liquidity info
    df["bid_depth"] = 1000.0
    df["ask_depth"] = 1000.0
    df["total_liquidity"] = df["bid_depth"] + df["ask_depth"]
    # approximate spread
    df["spread_percentage"] = ((df["high"] - df["low"])/df["low"]) * 10000.0
    # approximate volatility
    df["volatility"] = df["close"].pct_change().rolling(4).std() * 100
    # approximate taker ratio
    df["taker_buy_ratio"] = (df["taker_base_volume"] / (df["volume"] + 1e-9)).clip(0, 1)
    
    df.index.name = "timestamp"
    return df

########################
# 5. Evaluate Model
########################

def evaluate_model_on_df(df, model, scaler, feature_cols):
    """
    Given a DataFrame (with 'entry_success' after compute_features),
    plus the model, scaler, and feature list,
    compute and return the accuracy.
    """
    # Filter out rows without valid label
    df.dropna(subset=["entry_success"], inplace=True)
    if df.empty:
        return None  # no valid data

    X = df[feature_cols].copy()
    y = df["entry_success"].copy()
    
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_tensor).squeeze()
        preds_bin = (preds >= 0.5).int().numpy()
    
    acc = accuracy_score(y, preds_bin) * 100.0
    return acc


########################
# 6. Main: Run 2022-2024, then split
########################

def main():
    symbol = "BATUSDT"
    model_path = "./models/BATUSDT_model.pth"
    
    # A) Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model_state = checkpoint["model_state"]
    saved_scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    
    stablecoins = ["TUSDUSDT", "USDCUSDT", "BUSDUSDT", "USDTUSDT"]
    is_stablecoin = symbol in stablecoins
    
    model = PricePredictor(len(feature_cols), is_stablecoin=is_stablecoin)
    model.load_state_dict(model_state)
    model.eval()
    
    # Basic info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {symbol} model.")
    print(f" - Number of parameters: {total_params}")
    print(" - Feature columns:", feature_cols)
    
    # B) Fetch data from 2022-01-01 to 2024-01-01
    df = fetch_2022_to_2024(symbol)
    print(f"\nFetched {len(df)} rows of 15-minute data from 2022 through 2023.")
    
    # Show raw columns
    print("\nRaw columns from API data:")
    print(df.columns.tolist())
    
    # C) Apply feature engineering
    df = compute_features(df)
    print("\nColumns after feature engineering:")
    print(df.columns.tolist())
    
    # D) Evaluate 2022, 2023, and combined
    #    - We'll slice df by year
    df_2022 = df.loc["2022-01-01":"2023-01-01"].copy()
    df_2023 = df.loc["2023-01-01":"2024-01-01"].copy()
    
    # Combined is entire 2022 -> 2024
    # (which we already have in df)
    
    print("\nEvaluating the model...")

    # 1) Entire dataset (2022-01-01 to 2024-01-01)
    acc_all = evaluate_model_on_df(df, model, saved_scaler, feature_cols)
    if acc_all is not None:
        print(f"Overall (2022 + 2023) Accuracy: {acc_all:.2f}%")
    else:
        print("No data for combined period after feature engineering.")
    
    # 2) 2022 only
    acc_2022 = evaluate_model_on_df(df_2022, model, saved_scaler, feature_cols)
    if acc_2022 is not None:
        print(f"2022 Accuracy: {acc_2022:.2f}%")
    else:
        print("No valid 2022 data after feature engineering.")
    
    # 3) 2023 only
    acc_2023 = evaluate_model_on_df(df_2023, model, saved_scaler, feature_cols)
    if acc_2023 is not None:
        print(f"2023 Accuracy: {acc_2023:.2f}%")
    else:
        print("No valid 2023 data after feature engineering.")


if __name__ == "__main__":
    main()
