import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv
from model_components import PricePredictor, CryptoDataset

class ModelTrainer:
    def __init__(self, data_dir='./data/symbols/', models_dir='./models/'):
        """
        Initialize the ModelTrainer with directory paths
        
        :param data_dir: Directory containing symbol data files
        :param models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Define feature columns
        self.feature_cols = [
            'depth_imbalance', 'total_liquidity',
            'volatility_15m', 'volume_momentum',
            'vwap_deviation', 'price_momentum_15m',
            'rsi', 'macd', 'bollinger_position'
        ]
        
        # Initialize scaler
        self.scaler = RobustScaler(quantile_range=(5, 95))
    
    def load_data(self, file_path):
        """
        Load and preprocess data from a JSON file
        
        :param file_path: Path to the JSON data file
        :return: Processed DataFrame
        """
        logging.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r') as file:
            raw_data = json.load(file)
        
        processed_data = []
        for day_data in raw_data:
            try:
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
            except Exception as e:
                logging.warning(f"Error processing day data: {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("No valid data was processed from the input file")
            
        df = pd.DataFrame(processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').sort_index()
    
    def calculate_features(self, df):
        """
        Calculate features for the model
        
        :param df: Input DataFrame
        :return: DataFrame with calculated features
        """
        # Market Structure
        df['depth_imbalance'] = (
            (df['bid_depth'] - df['ask_depth']) / 
            (df['bid_depth'] + df['ask_depth'])
        ).clip(-0.5, 0.5)
        df['total_liquidity'] = np.log1p(df['total_liquidity'])
        
        # Market Dynamics
        df['returns'] = df['close'].pct_change().clip(-0.05, 0.05)
        df['volatility_15m'] = df['returns'].rolling(4).std()
        df['volume_momentum'] = (
            df['volume'] / df['volume'].rolling(96).mean()
        ).clip(0.1, 10)
        
        # Price Action
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (
            (df['close'] - df['vwap']) / df['vwap']
        ).clip(-0.05, 0.05)
        df['price_momentum_15m'] = df['returns'].rolling(4).mean()
        
        # Technical Indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], _ = self.calculate_macd(df['close'])
        df['bollinger_position'] = self.calculate_bollinger_position(df['close'])
        
        # Target Variable
        df['future_return'] = df['close'].shift(-16) / df['close'] - 1
        df['entry_success'] = (df['future_return'] > 0.015).astype(int)
        
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    # Other calculation methods remain the same
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

    def train_model(self, symbol, file_path, start_date='2024-01-01', end_date='2024-12-26', epochs=500, batch_size=256):
        """Train model with improved parameters and monitoring"""
        logging.info(f"Training model for {symbol}")
        
        # Load data
        df = self.load_data(file_path)
        df = self.calculate_features(df)
        
        # Check if stablecoin
        is_stablecoin = symbol in ['TUSDUSDT', 'USDCUSDT', 'BUSDUSDT', 'USDTUSDT']
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Split data into train and validation sets
        train_data = df[(df.index >= start_date) & (df.index <= end_date)]
        train_size = int(len(train_data) * 0.8)
        train_set = train_data.iloc[:train_size]
        val_set = train_data.iloc[train_size:]
        
        # Prepare training data
        X_train = train_set[self.feature_cols]
        y_train = train_set['entry_success']
        X_val = val_set[self.feature_cols]
        y_val = val_set['entry_success']
        
        # Improved feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Calculate class weights
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        class_weights = torch.FloatTensor([1.0, pos_weight])
        
        # Create datasets
        train_dataset = CryptoDataset(X_train_scaled, y_train.values)
        val_dataset = CryptoDataset(X_val_scaled, y_val.values)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = PricePredictor(len(self.feature_cols), is_stablecoin=is_stablecoin)
        
        # Loss and optimizer
        criterion = nn.BCELoss(reduction='none')
        
        if is_stablecoin:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            def lr_lambda(epoch):
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                return 0.5 ** (epoch / 50)
                
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop (simplified from original)
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience_counter = 0
        metrics_history = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y.unsqueeze(1))
                weights = class_weights[batch_y.long()]
                loss = (loss * weights.unsqueeze(1)).mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted.squeeze() == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1)).mean()
                    total_val_loss += loss.item()
                    
                    predicted = (outputs.data > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted.squeeze() == batch_y).sum().item()
            
            # Metrics calculation
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Logging and early stopping logic
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy
            })
            
            # Save best model
            if avg_val_loss < best_val_loss or val_accuracy > best_val_accuracy:
                best_val_loss = min(best_val_loss, avg_val_loss)
                best_val_accuracy = max(best_val_accuracy, val_accuracy)
                
                # Save model checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_cols': self.feature_cols,
                    'scaler': self.scaler
                }, self.models_dir / f'{symbol}_model.pth')
                
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 50:
                logging.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Final logging
        best_metrics = max(metrics_history, key=lambda x: x['val_acc'])
        logging.info(
            f'Final metrics for {symbol}: '
            f'Best epoch: {best_metrics["epoch"]}, '
            f'Train Loss: {best_metrics["train_loss"]:.4f}, '
            f'Val Loss: {best_metrics["val_loss"]:.4f}, '
            f'Train Acc: {best_metrics["train_acc"]:.2f}%, '
            f'Val Acc: {best_metrics["val_acc"]:.2f}%'
        )

def main():
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Discover available symbol files
    symbol_files = list(trainer.data_dir.glob('*USDT_*.json'))
    
    # Train models for all discovered symbols
    for file_path in symbol_files:
        symbol = file_path.stem.split('_')[0].upper()
        try:
            trainer.train_model(symbol, file_path)
        except Exception as e:
            logging.error(f"Error training model for {symbol}: {str(e)}")

if __name__ == "__main__":
    main()