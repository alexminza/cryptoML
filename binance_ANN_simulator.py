# binance_nn_trader.py
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class ETHDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ETHNNTrader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.model = None
        self.setup_logging()
        
        self.feature_cols = [
            'depth_imbalance', 'total_liquidity',
            'volatility_15m', 'volume_momentum',
            'vwap_deviation', 'price_momentum_15m',
            'rsi', 'macd', 'bollinger_position'
        ]

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}")
        with open(self.file_path, 'r') as file:
            raw_data = json.load(file)
        
        processed_data = []
        for day_data in raw_data:
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
                processed_data.append(record)
        
        df = pd.DataFrame(processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').sort_index()

    def calculate_features(self, df):
        # Market Structure
        df['depth_imbalance'] = np.clip((df['bid_depth'] - df['ask_depth']) / 
                                      (df['bid_depth'] + df['ask_depth']), -0.5, 0.5)
        df['total_liquidity'] = np.log1p(df['total_liquidity'])
        
        # Market Dynamics
        df['returns'] = df['close'].pct_change().clip(-0.05, 0.05)
        df['volatility_15m'] = df['returns'].rolling(4).std()
        df['volume_momentum'] = (df['volume'] / df['volume'].rolling(96).mean()).clip(0.1, 10)
        
        # Price Action
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = ((df['close'] - df['vwap']) / df['vwap']).clip(-0.05, 0.05)
        df['price_momentum_15m'] = df['returns'].rolling(4).mean()
        
        # Technical Indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], _ = self.calculate_macd(df['close'])
        df['bollinger_position'] = self.calculate_bollinger_position(df['close'])
        
        # Target Variable
        df['future_return'] = df['close'].shift(-16) / df['close'] - 1
        df['entry_success'] = (df['future_return'] > 0.01).astype(int)
        
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def calculate_bollinger_position(self, prices, window=20):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position.clip(0, 1)

    def train_model(self, start_date, end_date, epochs=100, batch_size=64):
        df = self.load_data()
        df = self.calculate_features(df)
        train_data = df[(df.index >= start_date) & (df.index <= end_date)]
        
        X = train_data[self.feature_cols]
        y = train_data['entry_success']
        
        X_scaled = self.scaler.fit_transform(X)
        
        dataset = ETHDataset(X_scaled, y.values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = PricePredictor(len(self.feature_cols))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def generate_signals(self, data):
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X = data[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled)).numpy()
        
        signals = data[
            (predictions.flatten() > 0.7) &
            (data['volume_momentum'] > 1.0) &
            (data['rsi'] < 70) &  # Not overbought
            (data['bollinger_position'] < 0.8)  # Not at upper band
        ].copy()
        
        signals['entry_price'] = signals['close'] * 0.995  # Limit order 0.5% below spot
        signals['target_price'] = signals['close'] * 1.01   # Take profit at 1%
        signals['stop_price'] = signals['entry_price'] * 0.995  # Stop loss at 0.5%
        
        logging.info(f"Generated {len(signals)} trading signals")
        return signals

    def simulate_trades(self, data, signals):
        trades = []
        for idx in range(len(signals)):
            timestamp = signals.index[idx]
            future_data = data.loc[timestamp:].iloc[1:16]
            
            if len(future_data) == 0:
                continue
                
            entry_price = signals.iloc[idx]['entry_price']
            target_price = signals.iloc[idx]['target_price']
            stop_price = signals.iloc[idx]['stop_price']
            
            exit_price, exit_time, reason = self.find_exit_point(
                entry_price, target_price, stop_price, future_data)
            
            trades.append({
                'entry_time': timestamp,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': (exit_price - entry_price) / entry_price,
                'exit_reason': reason
            })
        
        return pd.DataFrame(trades)

    def find_exit_point(self, entry_price, target_price, stop_price, future_data):
        for timestamp, row in future_data.iterrows():
            if row['low'] <= entry_price:  # Limit order filled
                if row['high'] >= target_price:
                    return target_price, timestamp, 'target'
                if row['low'] <= stop_price:
                    return stop_price, timestamp, 'stop'
        return future_data.iloc[-1]['close'], future_data.index[-1], 'timeout'

    def calculate_performance(self, trades, initial_capital=15000):
        if len(trades) == 0:
            return {}
            
        trades['win'] = trades['profit_pct'] > 0
        trades['streak'] = (trades['win'] != trades['win'].shift(1)).cumsum()
        max_losses = max([len(g) for _, g in trades.groupby('streak') if not g['win'].iloc[0]], default=0)
        
        trades['duration'] = pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])
        avg_duration_minutes = trades['duration'].mean().total_seconds() / 60
        
        # Calculate compound returns
        trades['position_size'] = initial_capital * 0.95  # Keep 5% as buffer
        trades['profit_amount'] = trades['position_size'] * trades['profit_pct']
        trades['cumulative_capital'] = initial_capital + trades['profit_amount'].cumsum()
        
        final_capital = trades['cumulative_capital'].iloc[-1]
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
        
        returns = trades['profit_pct']
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        performance = {
            'total_trades': len(trades),
            'winning_trades': len(trades[trades['profit_pct'] > 0]),
            'avg_profit': trades['profit_pct'].mean() * 100,
            'win_rate': len(trades[trades['profit_pct'] > 0]) / len(trades) * 100,
            'profit_factor': abs(
                trades[trades['profit_pct'] > 0]['profit_pct'].sum() /
                trades[trades['profit_pct'] < 0]['profit_pct'].sum()
            ) if len(trades[trades['profit_pct'] < 0]) > 0 else float('inf'),
            'max_consecutive_losses': max_losses,
            'avg_duration_minutes': avg_duration_minutes,
            'sharpe_ratio': sharpe,
            'total_return_pct': total_return_pct,
            'max_drawdown': self.calculate_drawdown(trades['profit_pct']),
            'initial_capital': initial_capital,
            'final_capital': final_capital
        }
        return performance

    def calculate_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdowns = cumulative / running_max - 1
        return abs(drawdowns.min()) * 100

    def backtest(self, start_date, end_date):
        df = self.load_data()
        df = self.calculate_features(df)
        period_data = df[(df.index >= start_date) & (df.index <= end_date)].copy()
        
        if len(period_data) == 0:
            logging.warning(f"No data found between {start_date} and {end_date}")
            return {}
            
        signals = self.generate_signals(period_data)
        
        if len(signals) == 0:
            logging.warning("No trading signals generated")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'avg_profit': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_consecutive_losses': 0,
                'avg_duration_minutes': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
            
        trades = self.simulate_trades(period_data, signals)
        performance = self.calculate_performance(trades)
        
        logging.info(f"Generated {len(signals)} signals, executed {len(trades)} trades")
        return performance

if __name__ == "__main__":
    trader = ETHNNTrader('./data/symbols/ETHUSDT_20241226.JSON')
    
    tinitial_capital = 15000
    
    try:
        print("\nTraining model on H1 2024...")
        trader.train_model('2024-01-01', '2024-06-30')
        
        print("\nBacktesting on H2 2024...")
        backtest_results = trader.backtest('2024-07-01', '2024-12-26')
        
        if backtest_results:
            print("\nBacktest Results:")
            print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
            print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
            print(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
            print(f"Total Trades: {backtest_results['total_trades']}")
            print(f"Winning Trades: {backtest_results['winning_trades']}")
            print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
            print(f"Average Profit per Trade: {backtest_results['avg_profit']:.2f}%")
            print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.2f}%")
        else:
            print("\nNo trading results generated")
            
    except Exception as e:
        print(f"Error: {str(e)}")