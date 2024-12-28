#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

########################
# 1. Define Your Model
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
# 2. Feature Engineering
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
# 3. Main Inspection Logic
########################

def inspect_and_test_model(
    model_path: str,
    json_data_path: str,
    symbol: str = "ETHUSDT"
):
    # A) Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model_state = checkpoint["model_state"]
    saved_scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    
    stablecoins = ["TUSDUSDT", "USDCUSDT", "BUSDUSDT", "USDTUSDT"]
    is_stablecoin = symbol in stablecoins
    
    model = PricePredictor(len(feature_cols), is_stablecoin=is_stablecoin)
    model.load_state_dict(model_state)
    model.eval()
    
    # B) Print Basic Model Info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {total_params}")
    print("Feature columns:", feature_cols)
    
    print("\nModel parameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  - {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")
    
    if hasattr(saved_scaler, "center_") and hasattr(saved_scaler, "scale_"):
        print("\nScaler center (from checkpoint):", saved_scaler.center_)
        print("Scaler scale (from checkpoint):", saved_scaler.scale_)
    
    # C) Load JSON Data
    with open(json_data_path, "r") as file:
        raw_data = json.load(file)
    
    records = []
    for day_data in raw_data:
        for price_data in day_data["prices_15min"]:
            record = {
                "timestamp": price_data["timestamp"],
                "open": float(price_data["open"]),
                "high": float(price_data["high"]),
                "low": float(price_data["low"]),
                "close": float(price_data["close"]),
                "volume": float(price_data["volume"]),
                "quote_volume": float(price_data["quote_volume"]),
                "bid_depth": float(day_data["liquidity"]["bid_depth"]),
                "ask_depth": float(day_data["liquidity"]["ask_depth"]),
                "total_liquidity": float(day_data["liquidity"]["total_liquidity"]),
                "spread_percentage": float(day_data["liquidity"]["spread_percentage"]),
                "volatility": float(day_data["additional_metrics"]["volatility"]),
                "taker_ratio": float(day_data["additional_metrics"]["taker_buy_ratio"])
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Show the raw columns
    print("\nTraining data - raw")
    print(list(df.columns))
    
    # D) Apply Feature Engineering
    df = compute_features(df)
    
    print("\nFeature engineering\n")
    print(list(df.columns))
    
    # Drop rows without a valid label
    df.dropna(subset=["entry_success"], inplace=True)
    if df.empty:
        print("\nNo data after feature engineering. Cannot test accuracy.")
        return
    
    # E) Scale & Predict
    X = df[feature_cols].copy()
    y = df["entry_success"].copy()
    
    X_scaled = saved_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_tensor).squeeze()
        preds_bin = (preds >= 0.5).int().numpy()
    
    accuracy = accuracy_score(y, preds_bin) * 100.0
    print(f"\nModel Accuracy on {symbol} data: {accuracy:.2f}%")


########################
# 4. Main Execution
########################

def main():
    # Update these paths to your actual model file and JSON data
    model_path = "./models/ETHUSDT_model.pth"
    json_data_path = "./data/symbols/ETHUSDT_20241226.json"
    symbol = "ETHUSDT"
    
    inspect_and_test_model(model_path, json_data_path, symbol)

if __name__ == "__main__":
    main()
