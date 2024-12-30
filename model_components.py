import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from binance.client import Client
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

def verify_data_directory():
    """Verify data directory structure and files exist"""
    data_dir = Path('./data/symbols/')
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created data directory: {data_dir}")
        return False
        
    json_files = list(data_dir.glob('*.json'))
    if not json_files:
        print(f"❌ No JSON files found in {data_dir}")
        print("Please ensure your data files are in the correct format: SYMBOLUSDT_YYYYMMDD.JSON")
        return False
        
    print("\nFound data files:")
    for file in json_files:
        print(f"✅ {file.name}")
    
    return True

class PricePredictor(nn.Module):
    VERSION = 3
    
    def __init__(self, input_size, is_stablecoin=False):
        super().__init__()
        
        if is_stablecoin:
            # Simpler architecture for stablecoins
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
            # Enhanced architecture for regular cryptocurrencies
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

class CryptoDataset(Dataset):
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

    def save_model(self, symbol):
        if self.model is None:
            return
            
        model_dir = Path('./train_models')
        model_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, model_dir / f'{symbol}_model.pth')
        
    def load_model(self, symbol):
        model_path = Path(f'./train_models/{symbol}_model.pth')
        if not model_path.exists():
            return False
            
        checkpoint = torch.load(model_path)
        self.model = PricePredictor(len(self.feature_cols))
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.feature_cols = checkpoint['feature_cols']
        return True
    
    def train_model(self, symbol, start_date, end_date, epochs=500, batch_size=256):
        """Train model with improved parameters and monitoring"""
        if self.load_model(symbol):
            logging.info(f"Loaded existing model for {symbol}")
            return
            
        logging.info(f"Training new model for {symbol}")
        df = self.load_data()
        df = self.calculate_features(df)
        
        # Check if stablecoin
        is_stablecoin = symbol in ['TUSDUSDT', 'USDCUSDT', 'BUSDUSDT', 'USDTUSDT']
        if is_stablecoin:
            logging.info(f"Using simplified model for stablecoin {symbol}")
        
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Split data into train and validation sets (80-20 split)
        train_data = df[(df.index >= start_date) & (df.index <= end_date)]
        train_size = int(len(train_data) * 0.8)
        train_set = train_data.iloc[:train_size]
        val_set = train_data.iloc[train_size:]
        
        if train_data.empty:
            raise ValueError(f"No training data found between {start_date} and {end_date}")
        
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
        
        # Initialize model and training components
        self.model = PricePredictor(len(self.feature_cols), is_stablecoin=is_stablecoin)
        
        # Use weighted BCE loss for imbalanced classes
        criterion = nn.BCELoss(reduction='none')
        
        # Use different parameters for stablecoins
        if is_stablecoin:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Create custom scheduler with warmup
            def lr_lambda(epoch):
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                return 0.5 ** (epoch / 50)  # Decay learning rate gradually
                
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience = 50 if not is_stablecoin else 20  # More patience for regular coins
        min_epochs = 100 if not is_stablecoin else 30  # Minimum training epochs
        patience_counter = 0
        
        metrics_history = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Apply class weights to loss
                loss = criterion(outputs, batch_y.unsqueeze(1))
                weights = class_weights[batch_y.long()]
                loss = (loss * weights.unsqueeze(1)).mean()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
                
                predicted = (outputs.data > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted.squeeze() == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss = loss.mean()
                    total_val_loss += loss.item()
                    
                    predicted = (outputs.data > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted.squeeze() == batch_y).sum().item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            if not is_stablecoin:
                scheduler.step()
            else:
                scheduler.step(avg_val_loss)
            
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy
            })
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                    f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%'
                )
            
            # Save best model based on both loss and accuracy
            if epoch >= min_epochs and (avg_val_loss < best_val_loss or val_accuracy > best_val_accuracy):
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model(symbol)
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= min_epochs:
                    logging.info(f'Early stopping triggered at epoch {epoch+1}')
                    break
        
        # Print final metrics and training summary
        best_metrics = max(metrics_history, key=lambda x: x['val_acc'])
        logging.info(
            f'Final metrics for {symbol}: '
            f'Best epoch: {best_metrics["epoch"]}, '
            f'Train Loss: {best_metrics["train_loss"]:.4f}, '
            f'Val Loss: {best_metrics["val_loss"]:.4f}, '
            f'Train Acc: {best_metrics["train_acc"]:.2f}%, '
            f'Val Acc: {best_metrics["val_acc"]:.2f}%'
        )

    def analyze_market_conditions(self, data):
        if data.empty:
            raise ValueError("No data available for market analysis")
            
        latest = data.iloc[-1]
        conditions = {
            'Volume Momentum': {
                'value': latest['volume_momentum'],
                'threshold': 0.8,
                'met': latest['volume_momentum'] > 0.8,
                'description': 'Trading volume compared to 24h average'
            },
            'RSI': {
                'value': latest['rsi'],
                'threshold': 75,
                'met': latest['rsi'] < 75,
                'description': 'Relative Strength Index (not overbought)'
            },
            'Bollinger Position': {
                'value': latest['bollinger_position'],
                'threshold': 0.85,
                'met': latest['bollinger_position'] < 0.85,
                'description': 'Position within Bollinger Bands'
            },
            'VWAP Deviation': {
                'value': latest['vwap_deviation'],
                'threshold': 0.02,
                'met': abs(latest['vwap_deviation']) < 0.02,
                'description': 'Price deviation from VWAP'
            },
            'Volatility': {
                'value': latest['volatility_15m'],
                'threshold': 0.03,
                'met': latest['volatility_15m'] < 0.03,
                'description': '15-minute volatility'
            }
        }
        return conditions

    def generate_signals(self, data):
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if data.empty:
            raise ValueError("No data available for signal generation")
            
        X = data[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled)).numpy()
        
        data['prediction'] = predictions.flatten()
        
        market_conditions = self.analyze_market_conditions(data)
        
        signals = data[
            (predictions.flatten() > 0.6) &
            (data['volume_momentum'] > 0.8) &
            (data['rsi'] < 75) &
            (data['bollinger_position'] < 0.85)
        ].copy()
        
        if not signals.empty:
            signals['entry_price'] = signals['close'] * 0.995
            signals['target_price'] = signals['close'] * 1.015
            signals['stop_price'] = signals['entry_price'] * 0.99
            
            signals['market_conditions'] = str(market_conditions)
            signals['conditions_met'] = sum(1 for cond in market_conditions.values() if cond['met'])
            signals['total_conditions'] = len(market_conditions)
        
        logging.info(f"Generated {len(signals)} trading signals")
        return signals, market_conditions

class MultiPairTrader:
    def __init__(self, symbols=None):  
        self.symbols = symbols  
        self.traders = {}
        self.client = None
        
        # Setup logging first
        self.setup_logging()
        
        # Setup Binance client
        try:
            self.setup_binance()
            print("✅ Successfully connected to Binance API")
        except Exception as e:
            print(f"\n❌ Error setting up Binance client: {str(e)}")
            print("Please check your .env file contains BINANCE_API_KEY and BINANCE_API_SECRET")
            return
        
        # Initialize directories and discover symbols
        self.init_directories()
        
    def init_directories(self):
        """Initialize required directories and check for data files"""
        data_dir = Path('./data/symbols/')
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created data directory: {data_dir}")
            return
        
        # Find all available symbol files
        json_files = list(data_dir.glob('*USDT_*.json'))
        if not json_files:
            print(f"❌ No JSON files found in {data_dir}")
            return
        
        # Extract symbols from available files
        available_symbols = set()
        for file in json_files:
            symbol = file.stem.split('_')[0].upper()  # Ensure uppercase
            if symbol.endswith('USDT'):
                available_symbols.add(symbol)
        
        if self.symbols is None:
            # If no symbols were specified, use all available ones
            self.symbols = sorted(list(available_symbols))
        else:
            # If symbols were specified, only use those that are available
            self.symbols = [s for s in self.symbols if s in available_symbols]
        
        print("\nFound data files for trading:")
        for symbol in self.symbols:
            print(f"✅ {symbol}")
            
    def initialize_traders(self):
        print("\nInitializing traders...")
        initialization_failed = False
        
        for symbol in self.symbols:
            # Try both date formats
            file_path = None
            for date_suffix in ['20241226', '20241227','20241228', '20241229', '20241230', '20241231']:
                temp_path = Path(f'./data/symbols/{symbol}_{date_suffix}.json')
                if temp_path.exists():
                    file_path = temp_path
                    break
            
            if file_path is None:
                print(f"❌ Data file not found for {symbol}")
                continue
            
            try:
                print(f"Initializing {symbol} trader...")
                self.traders[symbol] = ETHNNTrader(file_path)
                print(f"✅ Successfully initialized {symbol} trader")
            except Exception as e:
                print(f"❌ Error initializing {symbol} trader: {str(e)}")
                initialization_failed = True
        
        if not self.traders:
            print("\n❌ No traders were successfully initialized")
            return False
        elif initialization_failed:
            print("\n⚠️  Some traders failed to initialize, but others were successful")
        else:
            print("\n✅ All traders initialized successfully")
        
        print(f"\nTotal number of active traders: {len(self.traders)}")
        return len(self.traders) > 0
            
    def setup_binance(self):
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in .env file")
            
        self.client = Client(api_key, api_secret)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
            
    def train_all_models(self, start_date='2024-01-01', end_date='2024-12-26'):
        for symbol, trader in self.traders.items():
            logging.info(f"Training model for {symbol}...")
            try:
                trader.train_model(symbol, start_date, end_date)
            except Exception as e:
                logging.error(f"Error training model for {symbol}: {str(e)}")
            
    def get_current_market_data(self):
        market_data = {}
        for symbol in self.symbols:
            try:
                # Get recent klines
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_15MINUTE,
                    limit=100
                )
                
                # Get order book data
                depth = self.client.get_order_book(symbol=symbol)
                bid_depth = sum(float(bid[1]) for bid in depth['bids'][:10])
                ask_depth = sum(float(ask[1]) for ask in depth['asks'][:10])
                spread = (float(depth['asks'][0][0]) - float(depth['bids'][0][0])) / float(depth['bids'][0][0])
                
                # Create DataFrame from klines
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                    'buy_quote_volume', 'ignore'
                ])
                
                # Add order book metrics
                df['bid_depth'] = bid_depth
                df['ask_depth'] = ask_depth
                df['total_liquidity'] = bid_depth + ask_depth
                df['spread_percentage'] = spread * 100
                
                # Add volatility
                df['close'] = df['close'].astype(float)
                df['volatility'] = df['close'].pct_change().rolling(4).std() * 100
                
                # Convert timestamp and set index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                
                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                for col in numeric_cols:
                    df[col] = df[col].astype(float)
                    
                market_data[symbol] = df
                
            except Exception as e:
                logging.error(f"Error getting market data for {symbol}: {str(e)}")
                continue
                
        return market_data

    def generate_trading_signals(self):
            current_data = self.get_current_market_data()
            signals = {}
            
            for symbol, trader in self.traders.items():
                try:
                    data = current_data[symbol]
                    if data.empty:
                        logging.warning(f"No market data available for {symbol}")
                        continue
                        
                    processed_data = trader.calculate_features(data)
                    signal, market_conditions = trader.generate_signals(processed_data)
                    
                    try:
                        current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception as e:
                        logging.error(f"Error getting current price for {symbol}: {str(e)}")
                        current_price = float(processed_data['close'].iloc[-1])
                    
                    signal_data = {
                        'current_price': current_price,
                        'entry_price': current_price * 0.995,
                        'target_price': current_price * 0.995 * 1.015,
                        'stop_price': current_price * 0.995 * 0.99,
                        'market_conditions': market_conditions,
                        'signal_found': not signal.empty if isinstance(signal, pd.DataFrame) else False,
                    }
                    
                    if isinstance(signal, pd.DataFrame) and not signal.empty:
                        signal_data['prediction'] = float(signal['prediction'].iloc[-1])
                    else:
                        signal_data['prediction'] = 0
                        
                    signals[symbol] = signal_data
                    print(f"✅ Generated signals for {symbol}")
                        
                except Exception as e:
                    logging.error(f"Error generating signals for {symbol}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            return signals

def print_market_analysis(symbol, conditions):
    print(f"\n{symbol} Market Analysis:")
    print("=" * 50)
    conditions_met = sum(1 for cond in conditions.values() if cond['met'])
    print(f"Conditions Met: {conditions_met}/{len(conditions)}")
    
    print("\nDetailed Conditions:")
    print("-" * 50)
    for name, condition in conditions.items():
        status = "✅" if condition['met'] else "❌"
        print(f"{name}:")
        print(f"  Current: {condition['value']:.4f}")
        print(f"  Threshold: {condition['threshold']}")
        print(f"  Status: {status}")
        print(f"  Description: {condition['description']}")
        print()

def main():
    print("\n=== Crypto Trading Bot Initialization ===")
    print("\nStep 1: Verifying Environment")
    
    # Check for .env file
    if not Path('.env').exists():
        print("❌ .env file not found")
        print("Please create a .env file with your Binance API credentials:")
        print("BINANCE_API_KEY=your_api_key")
        print("BINANCE_API_SECRET=your_api_secret")
        return
    
    # Initialize traders with no predefined symbols to use all available ones
    print("\nStep 2: Initializing Traders")
    trader = MultiPairTrader(symbols=None) 
    
    if not trader.client:
        print("\n❌ Failed to initialize Binance client. Please fix the API credentials and try again.")
        return
    
    if not trader.initialize_traders():
        print("\n❌ Failed to initialize any traders. Please check the errors above and try again.")
        return
    
    print("\nStep 3: Training Models")
    print(f"\nTraining models for {len(trader.symbols)} pairs...")
    trader.train_all_models()
    
    print("\nStep 4: Analyzing Market Conditions")
    try:
        individual_signals = trader.generate_trading_signals()
        
        print("\nTrading Analysis Results:")
        analyzed_pairs = 0
        found_signals = 0
        
        for symbol in trader.symbols:
            individual_signal = individual_signals.get(symbol, {})
            if not individual_signal:
                print(f"\n❌ No signal data available for {symbol}")
                continue
            
            analyzed_pairs += 1
            market_conditions = individual_signal.get('market_conditions', {})
            if market_conditions:
                print_market_analysis(symbol, market_conditions)
                
                individual_confidence = individual_signal.get('prediction', 0)
                
                if individual_confidence > 0.6:
                    found_signals += 1
                    print(f"\n🎯 Strong Trading Signal Generated for {symbol}:")
                    print(f"Current Price: ${individual_signal['current_price']:.2f}")
                    print(f"Suggested Entry (Limit): ${individual_signal['entry_price']:.2f}")
                    print(f"Target Exit (Limit): ${individual_signal['target_price']:.2f}")
                    print(f"Stop Loss: ${individual_signal['stop_price']:.2f}")
                    print(f"Model Confidence: {individual_confidence:.2f}")
                    print(f"Potential Profit: {((individual_signal['target_price'] - individual_signal['entry_price']) / individual_signal['entry_price'] * 100):.2f}%")
                else:
                    print(f"\n❌ No strong trading signal for {symbol} - Confidence too low")
                    print(f"Model Confidence: {individual_confidence:.2f}")
            else:
                print(f"\n❌ Could not analyze market conditions for {symbol}")
        
        print(f"\n=== Trading Summary ===")
        print(f"Total Pairs Analyzed: {analyzed_pairs}")
        print(f"Strong Signals Found: {found_signals}")
        
        if found_signals > 0:
            print("\n=== Tradable Pairs Details ===")
            for symbol in trader.symbols:
                individual_signal = individual_signals.get(symbol, {})
                if not individual_signal:
                    continue
                
                individual_confidence = individual_signal.get('prediction', 0)
                
                if individual_confidence > 0.85:
                    market_conditions = individual_signal.get('market_conditions', {})
                    
                    print(f"\n{symbol} - Trading Opportunity")
                    print("-" * 40)
                    print(f"🔍 Market Conditions:")
                    conditions_met = 0
                    total_conditions = 0
                    for condition, details in market_conditions.items():
                        total_conditions += 1
                        status = "✅" if details.get('met', False) else "❌"
                        conditions_met += details.get('met', False)
                        print(f"   {condition.replace('_', ' ').title()}: {status}")
                        print(f"     Value: {details.get('value', 'N/A'):.4f}")
                        print(f"     Threshold: {details.get('threshold', 'N/A')}")
                        print(f"     Description: {details.get('description', 'N/A')}")
                    
                    print(f"\n📊 Trading Performance:")
                    print(f"   Conditions Met: {conditions_met}/{total_conditions}")
                    
                    print(f"\n💹 Trading Parameters:")
                    print(f"   Current Price: ${individual_signal['current_price']:.2f}")
                    print(f"   Entry Price (Limit): ${individual_signal['entry_price']:.2f}")
                    print(f"   Target Exit Price: ${individual_signal['target_price']:.2f}")
                    print(f"   Stop Loss Price: ${individual_signal['stop_price']:.2f}")
                    
                    print(f"\n🧠 Model Insights:")
                    print(f"   Market Entry Signal: {'Positive' if individual_confidence > 0.6 else 'Negative'}")
    
    except Exception as e:
        print(f"\n❌ Error during market analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()