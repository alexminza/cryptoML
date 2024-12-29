import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
import torch.serialization
import os

from utils.data_processor import DataProcessor
from utils.market_analysis import MarketAnalyzer
from train_models.price_predictor import PricePredictor

@dataclass
class TradeSignal:
    """Data class to hold trading signal information"""
    current_price: float
    entry_price: float
    target_price: float
    stop_price: float
    prediction: float
    market_conditions: Dict
    signal_found: bool
    
    def calculate_risk_reward_ratio(self) -> float:
        """Calculate the risk/reward ratio for the trade"""
        potential_reward = self.target_price - self.entry_price
        potential_risk = self.entry_price - self.stop_price
        return potential_reward / potential_risk if potential_risk != 0 else 0

class TradeAnalyzer:
    """Service class responsible for analyzing market conditions and generating trade signals"""
    
    def __init__(self, client, config):
        """Initialize the trade analyzer with client and configuration
        
        Args:
            client: Binance client instance
            config: Configuration instance containing settings
        """
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.traders: Dict[str, DataProcessor] = {}
        self.models: Dict[str, PricePredictor] = {}
        self.market_analyzer = MarketAnalyzer(config)
        
    def initialize_traders(self) -> bool:
        """Initialize traders with loaded models"""
        self.logger.info("Initializing traders...")
        initialization_failed = False
        
        for file_path in self.config.DATA_DIR.glob('*USDT_*.json'):
            try:
                symbol = file_path.stem.split('_')[0].upper()
                
                # Initialize data processor
                self.traders[symbol] = DataProcessor(file_path)
                
                # Load trained model
                model_path = self.config.MODEL_DIR / f'{symbol}_model.pth'
                if not model_path.exists():
                    self.logger.warning(f"No trained model found for {symbol}")
                    continue
                    
                try:
                    self.models[symbol] = self._load_model(model_path)
                    self.logger.info(f"✅ Successfully initialized {symbol} trader")
                except Exception as e:
                    self.logger.error(f"❌ Error initializing {symbol} trader: {str(e)}")
                    initialization_failed = True
                    continue
                
            except Exception as e:
                self.logger.error(f"❌ Error initializing {symbol}: {str(e)}")
                initialization_failed = True
                continue
                
        if not self.traders:
            self.logger.error("No traders were successfully initialized")
            return False
            
        self.logger.info(f"Total number of active traders: {len(self.traders)}")
        return True

    def _load_model(self, model_path: Path) -> Optional[PricePredictor]:
        """
        Load a trained model from disk with security checks
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Optional[PricePredictor]: Loaded model or None if loading fails
            
        Raises:
            ValueError: If model path is outside trusted directory or file doesn't exist
        """
        try:
            # Security checks
            model_path = Path(model_path).resolve()  # Resolve any symlinks
            trusted_dir = Path(self.config.MODEL_DIR).resolve()
            
            # Check if model file exists
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
                
            # Check if model is within trusted directory
            if not str(model_path).startswith(str(trusted_dir)):
                raise ValueError(f"Model path {model_path} is outside of trusted models directory")
                
            # Check file extension
            if model_path.suffix.lower() != '.pth':
                raise ValueError(f"Invalid model file extension: {model_path.suffix}")
                
            # Check file size (e.g., max 1GB)
            max_size = 1024 * 1024 * 1024  # 1GB in bytes
            if os.path.getsize(model_path) > max_size:
                raise ValueError(f"Model file too large: {model_path}")
                
            # Check file permissions (should be readable but not executable)
            file_mode = os.stat(model_path).st_mode
            if file_mode & 0o111:  # Check if any execute bits are set
                raise ValueError(f"Model file has executable permissions: {model_path}")
                
            # Load the model with CPU mapping
            checkpoint = torch.load(
                model_path,
                weights_only=False,  # We trust our own models
                map_location=torch.device('cpu')
            )
            
            # Validate checkpoint structure
            required_keys = {'model_state', 'feature_cols'}
            if not all(key in checkpoint for key in required_keys):
                raise ValueError(f"Invalid checkpoint format: missing required keys {required_keys}")
                
            # Initialize and load model
            feature_count = len(checkpoint['feature_cols'])
            if not isinstance(feature_count, int) or feature_count <= 0:
                raise ValueError(f"Invalid feature count: {feature_count}")
                
            model = PricePredictor(feature_count)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()  # Set to evaluation mode
            
            self.logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
        
    def _fetch_klines(self, symbol: str) -> List[List]:
        """Fetch kline (candlestick) data from Binance"""
        return self.client.get_klines(
            symbol=symbol,
            interval=self.client.KLINE_INTERVAL_15MINUTE,
            limit=100
        )
        
    def _fetch_order_book(self, symbol: str) -> Dict:
        """Fetch order book data from Binance"""
        return self.client.get_order_book(symbol=symbol)
        
    def _process_market_data(self, klines: List[List], depth: Dict) -> pd.DataFrame:
        """Process raw market data into a DataFrame"""
        # Create DataFrame from klines
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'buy_base_volume',
            'buy_quote_volume', 'ignore'
        ])
        
        # Add order book metrics
        bid_depth = sum(float(bid[1]) for bid in depth['bids'][:10])
        ask_depth = sum(float(ask[1]) for ask in depth['asks'][:10])
        spread = (float(depth['asks'][0][0]) - float(depth['bids'][0][0])) / float(depth['bids'][0][0])
        
        df['bid_depth'] = bid_depth
        df['ask_depth'] = ask_depth
        df['total_liquidity'] = bid_depth + ask_depth
        df['spread_percentage'] = spread * 100
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = df[col].astype(float)
            
        # Add volatility
        df['volatility'] = df['close'].pct_change().rolling(4).std() * 100
        
        # Convert timestamp and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp').sort_index()
        
    def get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch current market data for all symbols"""
        market_data = {}
        
        for symbol in self.traders:
            try:
                klines = self._fetch_klines(symbol)
                depth = self._fetch_order_book(symbol)
                
                processed_data = self._process_market_data(klines, depth)
                market_data[symbol] = processed_data
                
            except Exception as e:
                self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
                
        return market_data
        
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            self.logger.warning(f"Error getting current price for {symbol}: {str(e)}")
            return None
            
    def generate_trading_signals(self) -> Dict[str, TradeSignal]:
        """Generate trading signals for all symbols"""
        current_data = self.get_current_market_data()
        signals = {}
        
        for symbol, trader in self.traders.items():
            try:
                signal = self._generate_single_signal(symbol, trader, current_data)
                if signal:  # Only add valid signals
                    signals[symbol] = signal
                    self.logger.info(f"✅ Generated signals for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
                continue
                
        return signals
        
    def _generate_single_signal(
        self, symbol: str, trader: DataProcessor, current_data: Dict
    ) -> Optional[TradeSignal]:
        """Generate trading signal for a single symbol"""
        data = current_data.get(symbol)
        if data is None or data.empty:
            self.logger.warning(f"No market data available for {symbol}")
            return None
            
        try:
            processed_data = trader.calculate_features(data)
            model = self.models.get(symbol)
            
            if model is None:
                self.logger.warning(f"No model available for {symbol}")
                return None
                
            with torch.no_grad():
                features = processed_data[trader.feature_cols].iloc[-1:].values
                prediction = model(torch.FloatTensor(features)).item()
                
            market_conditions = self.market_analyzer.analyze_market_conditions(processed_data)
            current_price = self._get_current_price(symbol)
            
            if current_price is None:
                return None
                
            return TradeSignal(
                current_price=current_price,
                entry_price=current_price * self.config.ENTRY_PRICE_FACTOR,
                target_price=current_price * self.config.ENTRY_PRICE_FACTOR * self.config.TARGET_PRICE_FACTOR,
                stop_price=current_price * self.config.ENTRY_PRICE_FACTOR * self.config.STOP_LOSS_FACTOR,
                prediction=prediction,
                market_conditions=market_conditions,
                signal_found=prediction > self.config.CONFIDENCE_THRESHOLD
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
            
    def print_trade_opportunity(self, symbol: str, signal: TradeSignal):
        """Print detailed information about a trading opportunity"""
        print(f"\n{symbol} - Trading Opportunity")
        print("-" * 40)
        
        print(f"🔍 Market Conditions:")
        conditions_met = sum(1 for condition in signal.market_conditions.values() if condition.met)
        total_conditions = len(signal.market_conditions)
        
        for name, condition in signal.market_conditions.items():
            status = "✅" if condition.met else "❌"
            print(f"   {name}: {status}")
            print(f"     Current: {condition.value:.4f}")
            print(f"     Threshold: {condition.threshold}")
            
        print(f"\n📊 Signal Strength:")
        print(f"   Model Confidence: {signal.prediction:.2%}")
        print(f"   Conditions Met: {conditions_met}/{total_conditions}")
        
        print(f"\n💰 Trade Parameters:")
        print(f"   Current Price: ${signal.current_price:.2f}")
        print(f"   Entry Price: ${signal.entry_price:.2f}")
        print(f"   Target Price: ${signal.target_price:.2f}")
        print(f"   Stop Loss: ${signal.stop_price:.2f}")
        
        risk_reward = (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_price)
        print(f"   Risk/Reward Ratio: {risk_reward:.2f}")
        print(f"   Potential Profit: {((signal.target_price - signal.entry_price) / signal.entry_price * 100):.2f}%")
        print(f"   Maximum Risk: {((signal.entry_price - signal.stop_price) / signal.entry_price * 100):.2f}%")