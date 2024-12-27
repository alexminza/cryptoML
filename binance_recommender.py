import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_suggestions.log'),
        logging.StreamHandler()
    ]
)

class TradingSuggestions:
    def __init__(self, 
                 min_daily_volume_usdt: float = 10000000,  # 10M USDT minimum volume
                 volatility_lookback: int = 5,
                 num_pairs: int = 10):
        
        load_dotenv()
        self.client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
        self.min_daily_volume_usdt = min_daily_volume_usdt
        self.volatility_lookback = volatility_lookback
        self.num_pairs = num_pairs

    def get_historical_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        try:
            klines = self.client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1DAY,
                str(datetime.now() - timedelta(days=lookback_days))
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate trading metrics for a symbol"""
        try:
            # Calculate volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * 100
            
            # Calculate average daily volume
            avg_volume = df['quote_volume'].mean()
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Calculate suggested entry and exit prices
            entry_price = current_price * 0.995  # -0.5%
            exit_price = entry_price * 1.015     # +1.5% from entry
            
            # Calculate daily range
            daily_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100
            
            return {
                'volatility': volatility,
                'avg_volume': avg_volume,
                'current_price': current_price,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'daily_range': daily_range
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            return None

    def get_trading_suggestions(self):
        """Get trading suggestions for tomorrow"""
        try:
            # Get all USDT trading pairs
            exchange_info = self.client.get_exchange_info()
            all_symbols = [s['symbol'] for s in exchange_info['symbols'] 
                         if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
            
            logging.info(f"Analyzing {len(all_symbols)} trading pairs...")
            
            # Store metrics for each symbol
            symbol_metrics = []
            
            for symbol in all_symbols:
                df = self.get_historical_data(symbol, self.volatility_lookback + 1)
                if df is None or len(df) < self.volatility_lookback:
                    continue
                
                metrics = self.calculate_metrics(df)
                if metrics is None:
                    continue
                
                # Only consider symbols meeting volume threshold
                if metrics['avg_volume'] >= self.min_daily_volume_usdt:
                    symbol_metrics.append({
                        'symbol': symbol,
                        **metrics
                    })
            
            # Sort by combined rank of volatility and volume
            df_metrics = pd.DataFrame(symbol_metrics)
            df_metrics['volatility_rank'] = df_metrics['volatility'].rank(ascending=False)
            df_metrics['volume_rank'] = df_metrics['avg_volume'].rank(ascending=False)
            df_metrics['combined_rank'] = df_metrics['volatility_rank'] + df_metrics['volume_rank']
            
            # Select top pairs
            suggestions = df_metrics.nsmallest(self.num_pairs, 'combined_rank')
            
            # Prepare results
            results = []
            for _, row in suggestions.iterrows():
                results.append({
                    'symbol': row['symbol'],
                    'current_price': row['current_price'],
                    'suggested_entry': row['entry_price'],
                    'suggested_exit': row['exit_price'],
                    'daily_volatility': row['volatility'],
                    'daily_volume_usdt': row['avg_volume'],
                    'daily_range': row['daily_range'],
                    'potential_profit': ((row['exit_price'] / row['entry_price']) - 1) * 100
                })
            
            # Save suggestions to file
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            with open(f'trading_suggestions_{timestamp}.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'suggestions': results
                }, f, indent=4)
            
            return results
            
        except Exception as e:
            logging.error(f"Error getting trading suggestions: {e}")
            return []

def print_suggestions(suggestions):
    """Print trading suggestions in a formatted way"""
    print("\n=== Trading Suggestions for Tomorrow ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTop 10 Pairs Selected Based on Volume and Volatility:\n")
    
    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. {sugg['symbol']}")
        print(f"   Current Price: ${sugg['current_price']:.8f}")
        print(f"   Entry Price (-0.5%): ${sugg['suggested_entry']:.8f}")
        print(f"   Exit Price (+1.5%): ${sugg['suggested_exit']:.8f}")
        print(f"   24h Volume: ${sugg['daily_volume_usdt']:,.2f}")
        print(f"   Volatility: {sugg['daily_volatility']:.2f}%")
        print(f"   Daily Range: {sugg['daily_range']:.2f}%")
        print(f"   Potential Profit: {sugg['potential_profit']:.2f}%")

if __name__ == "__main__":
    analyzer = TradingSuggestions(
        min_daily_volume_usdt=10000000,
        volatility_lookback=5,
        num_pairs=10
    )
    
    suggestions = analyzer.get_trading_suggestions()
    print_suggestions(suggestions)