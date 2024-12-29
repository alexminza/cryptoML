import json
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict
from models.crypto_dataset import CryptoDataset

class DataProcessor:
    """Utility class for processing and preparing trading data"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = RobustScaler(quantile_range=(5, 95))
        
        self.feature_cols = [
            'depth_imbalance', 'total_liquidity',
            'volatility_15m', 'volume_momentum',
            'vwap_deviation', 'price_momentum_15m',
            'rsi', 'macd', 'bollinger_position'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """Load and process data from JSON file"""
        with open(self.file_path, 'r') as file:
            raw_data = json.load(file)
            
        processed_data = []
        for day_data in raw_data:
            try:
                processed_data.extend(self._process_day_data(day_data))
            except Exception as e:
                continue
                
        if not processed_data:
            raise ValueError("No valid data was processed from the input file")
            
        df = pd.DataFrame(processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').sort_index()
        
    def _process_day_data(self, day_data: Dict) -> list:
        """Process single day's worth of data"""
        processed_records = []
        
        for price_data in day_data['prices_15min']:
            record = {
                'timestamp': price_data['timestamp'],
                'open': float(price_data['open']),
                'high': float(price_data['high']),
                'low': float(price_data['low']),
                'close': float(price_data['close']),
                'volume': float(price_data['volume']),
                'quote_volume': float(price_data['quote_volume']),
                'bid_depth': float(day_data['liquidity']['bid_depth']),
                'ask_depth': float(day_data['liquidity']['ask_depth']),
                'total_liquidity': float(day_data['liquidity']['total_liquidity']),
                'spread_percentage': float(day_data['liquidity']['spread_percentage']),
                'volatility': float(day_data['additional_metrics']['volatility']),
                'taker_ratio': float(day_data['additional_metrics']['taker_buy_ratio'])
            }
            processed_records.append(record)
            
        return processed_records
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading features from raw data"""
        df = df.copy()
        
        # Market Structure
        df['depth_imbalance'] = self._calculate_depth_imbalance(df)
        df['total_liquidity'] = np.log1p(df['total_liquidity'])
        
        # Market Dynamics
        df['returns'] = df['close'].pct_change().clip(-0.05, 0.05)
        df['volatility_15m'] = df['returns'].rolling(4).std()
        df['volume_momentum'] = self._calculate_volume_momentum(df)
        
        # Price Action
        df = self._calculate_price_action_features(df)
        
        # Technical Indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], _ = self._calculate_macd(df['close'])
        df['bollinger_position'] = self._calculate_bollinger_position(df['close'])
        
        # Target Variable
        df['future_return'] = df['close'].shift(-16) / df['close'] - 1
        df['entry_success'] = (df['future_return'] > 0.015).astype(int)
        
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    def prepare_train_val_split(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> Tuple[CryptoDataset, CryptoDataset]:
        """Prepare training and validation datasets"""
        train_data = df[(df.index >= start_date) & (df.index <= end_date)]
        train_size = int(len(train_data) * 0.8)
        
        train_set = train_data.iloc[:train_size]
        val_set = train_data.iloc[train_size:]
        
        X_train = train_set[self.feature_cols]
        y_train = train_set['entry_success']
        X_val = val_set[self.feature_cols]
        y_val = val_set['entry_success']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return (
            CryptoDataset(X_train_scaled, y_train.values),
            CryptoDataset(X_val_scaled, y_val.values)
        )
        
    @staticmethod
    def _calculate_depth_imbalance(df: pd.DataFrame) -> pd.Series:
        """Calculate order book depth imbalance"""
        return (
            (df['bid_depth'] - df['ask_depth']) /
            (df['bid_depth'] + df['ask_depth'])
        ).clip(-0.5, 0.5)
        
    @staticmethod
    def _calculate_volume_momentum(df: pd.DataFrame) -> pd.Series:
        """Calculate volume momentum"""
        return (
            df['volume'] / df['volume'].rolling(96).mean()
        ).clip(0.1, 10)
        
    @staticmethod
    def _calculate_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related features"""
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (
            (df['close'] - df['vwap']) / df['vwap']
        ).clip(-0.05, 0.05)
        df['price_momentum_15m'] = df['returns'].rolling(4).mean()
        return df
        
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
        
    @staticmethod
    def _calculate_bollinger_position(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position.clip(0, 1)
        
    def save_model(self, model, symbol: str):
        """Save trained model and scaler"""
        torch.save({
            'model_state': model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, f'./models/{symbol}_model.pth')