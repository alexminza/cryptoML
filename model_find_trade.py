import os
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv

# Import necessary components
from model_training import ModelTrainer
from model_components import PricePredictor 

class TradeFinder:
    def __init__(self, models_dir='./models/', data_dir='./data/symbols/'):
        """
        Initialize TradeFinder
        
        :param models_dir: Directory containing trained models
        :param data_dir: Directory containing symbol data files
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Setup Binance client
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in .env file")
        
        self.client = Client(api_key, api_secret)
    
    def load_model(self, symbol, feature_cols):
        """
        Load a pre-trained model for a symbol
        
        :param symbol: Trading symbol
        :param feature_cols: Feature columns used in training
        :return: Loaded model, feature columns, and scaler
        """
        model_path = self.models_dir / f'{symbol}_model.pth'
        
        if not model_path.exists():
            logging.warning(f"No model found for {symbol}")
            return None, None, None
        
        try:
            # Print the contents of the checkpoint for debugging
            checkpoint = torch.load(model_path)
            
            # Debug print the checkpoint keys
            logging.info(f"Checkpoint keys for {symbol}: {checkpoint.keys()}")
            
            # Recreate model with the same architecture
            is_stablecoin = symbol in ['TUSDUSDT', 'USDCUSDT', 'BUSDUSDT', 'USDTUSDT']
            model = PricePredictor(len(feature_cols), is_stablecoin=is_stablecoin)
            
            # Check if 'model_state_dict' exists in checkpoint
            if 'model_state_dict' not in checkpoint:
                logging.error(f"No 'model_state_dict' found in checkpoint for {symbol}")
                return None, None, None
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            # Verify other required keys
            if 'feature_cols' not in checkpoint:
                logging.warning(f"No 'feature_cols' found in checkpoint for {symbol}. Using provided feature_cols.")
                checkpoint['feature_cols'] = feature_cols
            
            if 'scaler' not in checkpoint:
                logging.warning(f"No 'scaler' found in checkpoint for {symbol}")
                return None, None, None
            
            return model, checkpoint['feature_cols'], checkpoint['scaler']
        
        except Exception as e:
            logging.error(f"Error loading model for {symbol}: {str(e)}")
            # If possible, print the full traceback
            import traceback
            logging.error(traceback.format_exc())
            return None, None, None
    
    def get_current_market_data(self, symbol):
        """
        Retrieve current market data for a symbol
        
        :param symbol: Trading symbol
        :return: DataFrame with market data
        """
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
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
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
        
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    # Calculation methods (similar to those in ModelTrainer)
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
    
    def find_trading_opportunities(self):
        """
        Find trading opportunities across all symbols
        
        :return: Dictionary of trading opportunities
        """
        trading_opportunities = {}
        
        # Discover symbols from model files
        model_files = list(self.models_dir.glob('*_model.pth'))
        symbols = [file.stem.split('_')[0] for file in model_files]
        
        for symbol in symbols:
            try:
                # Get current market price
                current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                
                # Get market data
                market_data = self.get_current_market_data(symbol)
                
                if market_data is None:
                    continue
                
                # Calculate features
                processed_data = self.calculate_features(market_data)
                
                # Load model
                model_cols = [
                    'depth_imbalance', 'total_liquidity',
                    'volatility_15m', 'volume_momentum',
                    'vwap_deviation', 'price_momentum_15m',
                    'rsi', 'macd', 'bollinger_position'
                ]
                model, feature_cols, scaler = self.load_model(symbol, model_cols)
                
                if model is None:
                    continue
                
                # Prepare features
                X = processed_data[feature_cols]
                X_scaled = scaler.transform(X)
                
                # Generate prediction
                with torch.no_grad():
                    prediction = model(torch.FloatTensor(X_scaled))[-1].item()
                
                # Analyze trading opportunity
                if prediction > 0.6:
                    trading_opportunities[symbol] = {
                        'prediction': prediction,
                        'current_price': current_price,
                        'entry_price': current_price * 0.995,
                        'target_price': current_price * 1.015,
                        'stop_price': current_price * 0.99
                    }
            
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
        
        return trading_opportunities

def main():
    try:
        trade_finder = TradeFinder()
        opportunities = trade_finder.find_trading_opportunities()
        
        print("\n=== Trading Opportunities ===")
        if not opportunities:
            print("No trading opportunities found.")
            return
        
        for symbol, details in opportunities.items():
            print(f"\n{symbol} Trading Opportunity:")
            print(f"Current Price: ${details['current_price']:.2f}")
            print(f"Entry Price: ${details['entry_price']:.2f}")
            print(f"Target Price: ${details['target_price']:.2f}")
            print(f"Stop Price: ${details['stop_price']:.2f}")
            print(f"Entry Likelihood: {details['prediction']:.2%}")
    
    except Exception as e:
        logging.error(f"An error occurred during trade finding: {str(e)}")

if __name__ == "__main__":
    main()