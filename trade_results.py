import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from binance.client import Client
from dotenv import load_dotenv
import time
import json
import csv
import logging
import talib
from tqdm import tqdm
from typing import Dict, List, Optional, Union

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_binance_client():
    """Initialize Binance client with credentials from .env"""
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file")
    
    return Client(api_key, api_secret)

def load_symbol_data(checkpoint_file: Optional[str] = None, default_symbols: Optional[List[str]] = None) -> List[str]:
    """
    Load trading symbols from checkpoint JSON file or use default symbols.
    If both are provided, combines them without duplicates.
    """
    symbols = set(default_symbols or [])
    
    if checkpoint_file:
        try:
            # Try multiple possible locations
            possible_paths = [
                checkpoint_file,
                f"./Trader_Agent/{checkpoint_file}",
                f"./{checkpoint_file}",
                os.path.join(os.path.dirname(__file__), checkpoint_file)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        symbol_data = json.load(f)
                        symbols.update(symbol_data.keys())
                    logging.info(f"Loaded symbols from {path}")
                    break
            else:
                logging.warning(f"Checkpoint file not found in any location, using default symbols")
                
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in checkpoint file")
        except Exception as e:
            logging.error(f"Error loading checkpoint file: {str(e)}")
    
    # If no symbols were loaded, use some common trading pairs
    if not symbols:
        default_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'BCHUSDT', 'LTCUSDT',
            'LINKUSDT', 'XLMUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
        ]
        symbols.update(default_pairs)
        logging.info("Using default trading pairs")
    
    return list(symbols)

def get_market_data(client: Client, symbol: str, timestamp: Union[datetime, pd.Timestamp], 
                   interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """Fetch historical market data around the trade timestamp"""
    try:
        end_time = timestamp + timedelta(hours=1)
        start_time = timestamp - timedelta(hours=limit)
        
        klines = client.get_historical_klines(
            symbol,
            interval,
            int(start_time.timestamp() * 1000),
            int(end_time.timestamp() * 1000)
        )
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    except Exception as e:
        logging.error(f"Error fetching market data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical indicators for market data"""
    if len(df) < 100:
        return {}
    
    try:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        # RSI
        rsi = talib.RSI(close_prices, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Moving Averages
        ma7 = talib.SMA(close_prices, timeperiod=7)
        ma25 = talib.SMA(close_prices, timeperiod=25)
        ma50 = talib.SMA(close_prices, timeperiod=50)
        ma100 = talib.SMA(close_prices, timeperiod=100)
        
        # Other indicators
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        obv = talib.OBV(close_prices, volume)
        
        # VWAP calculation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Volume metrics
        volume_metrics = {
            f'volume_{period}h': df['volume'].rolling(window=int(period)).mean().iloc[-1]
            for period in [1, 4, 24]  # 1h, 4h, 1d
        }
        
        indicators = {
            'rsi': rsi[-1],
            'bb_upper': bb_upper[-1],
            'bb_middle': bb_middle[-1],
            'bb_lower': bb_lower[-1],
            'macd': macd[-1],
            'macd_signal': macd_signal[-1],
            'macd_hist': macd_hist[-1],
            'ma7': ma7[-1],
            'ma25': ma25[-1],
            'ma50': ma50[-1],
            'ma100': ma100[-1],
            'atr': atr[-1],
            'adx': adx[-1],
            'obv': obv[-1],
            'vwap': df['vwap'].iloc[-1],
            **volume_metrics
        }
        
        return {k: float(v) for k, v in indicators.items() if not np.isnan(v)}
    
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        return {}

def aggregate_related_trades(trades: List[Dict]) -> List[Dict]:
    """Aggregate related trades (e.g., partial fills, stop-loss executions)"""
    try:
        # Convert numeric fields to float before creating DataFrame
        for trade in trades:
            trade['qty'] = float(trade['qty'])
            trade['price'] = float(trade['price'])
            trade['commission'] = float(trade['commission'])
            
        trades_df = pd.DataFrame(trades)
        
        # Convert time to datetime, handling both millisecond timestamps and ISO format
        def convert_time(t):
            if isinstance(t, (int, float)):
                return pd.to_datetime(float(t), unit='ms')
            elif isinstance(t, str):
                return pd.to_datetime(t)
            return t
            
        trades_df['time'] = trades_df['time'].apply(convert_time)
        
        # Sort by time
        trades_df = trades_df.sort_values('time')
        
        # Group trades that occur within 1 second of each other
        trades_df['trade_group'] = (
            trades_df['time'].diff().dt.total_seconds() > 1
        ).cumsum()
        
        aggregated_trades = []
        for _, group in trades_df.groupby('trade_group'):
            if len(group) == 1:
                aggregated_trades.append(group.iloc[0].to_dict())
                continue
                
            # Aggregate multiple fills
            qty_sum = group['qty'].sum()
            agg_trade = {
                'symbol': group['symbol'].iloc[0],
                'time': group['time'].min(),
                'qty': qty_sum,
                'price': (group['price'] * group['qty']).sum() / qty_sum,
                'commission': group['commission'].sum(),
                'commissionAsset': group['commissionAsset'].iloc[0],
                'isBuyer': group['isBuyer'].iloc[0]
            }
            aggregated_trades.append(agg_trade)
        
        return aggregated_trades
    
    except Exception as e:
        logging.error(f"Error aggregating trades: {str(e)}")
        return trades

def calculate_trade_statistics(results: List[Dict]) -> Dict[str, float]:
    """Calculate comprehensive trading statistics"""
    try:
        stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'unprofitable_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'total_fees': 0.0,
            'total_buy_value': 0.0,
            'total_sell_value': 0.0,
            'total_buy_qty': 0.0,
            'total_sell_qty': 0.0,
            'hold_durations': []
        }
        
        for result in results:
            stats['total_buy_value'] += result['buy_value']
            stats['total_sell_value'] += result['sell_value']
            stats['total_buy_qty'] += result['buy_quantity']
            stats['total_sell_qty'] += result['sell_quantity']
            
            trades = result['trade_details']
            fees_per_trade = result['fees'] / len(trades) if trades else 0
            
            for trade in trades:
                net_pnl = trade['pnl'] - fees_per_trade
                stats['total_trades'] += 1
                
                if net_pnl > 0:
                    stats['profitable_trades'] += 1
                    stats['total_profit'] += net_pnl
                else:
                    stats['unprofitable_trades'] += 1
                    stats['total_loss'] += abs(net_pnl)
                
                stats['hold_durations'].append(trade['hold_duration'])
                stats['total_fees'] += fees_per_trade
        
        stats['success_rate'] = (
            (stats['profitable_trades'] / stats['total_trades'] * 100)
            if stats['total_trades'] > 0 else 0
        )
        
        stats['avg_hold_duration'] = (
            sum(stats['hold_durations']) / len(stats['hold_durations'])
            if stats['hold_durations'] else 0
        )
        
        stats['net_profit'] = stats['total_profit'] - stats['total_loss'] - stats['total_fees']
        
        return stats
    
    except Exception as e:
        logging.error(f"Error calculating statistics: {str(e)}")
        return {}

def get_trades_for_period(client: Client, symbol: str, 
                         start_time: datetime, end_time: datetime) -> List[Dict]:
    """Get trades for a specific symbol in given time period"""
    all_trades = []
    current_start = start_time
    chunk_count = 0
    
    try:
        while current_start < end_time:
            chunk_count += 1
            chunk_end = min(current_start + timedelta(hours=24), end_time)
            
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(chunk_end.timestamp() * 1000)
            
            chunk_trades = client.get_my_trades(
                symbol=symbol,
                startTime=start_ms,
                endTime=end_ms,
                limit=1000
            )
            
            if chunk_trades:
                all_trades.extend(chunk_trades)
            
            current_start = chunk_end
            time.sleep(0.1)  # Rate limiting
        
        return all_trades
        
    except Exception as e:
        if 'Invalid symbol' not in str(e):
            logging.error(f"Error fetching trades for {symbol}: {str(e)}")
        return []

def process_symbol_trades(client: Client, trades: List[Dict]) -> Optional[Dict]:
    """Process trades with market data collection and aggregation"""
    if not trades:
        return None
        
    try:
        symbol = trades[0]['symbol']
        
        # Convert timestamps in trades before aggregation
        for trade in trades:
            if isinstance(trade['time'], (int, float)):
                trade['time'] = int(trade['time'])
            elif isinstance(trade['time'], str):
                trade['time'] = int(datetime.fromisoformat(trade['time'].replace('Z', '+00:00')).timestamp() * 1000)
                
        aggregated_trades = aggregate_related_trades(trades)
        
        buy_queue = []
        trade_details = []
        
        # Initialize tracking
        matched_buy_value = matched_sell_value = matched_buy_qty = matched_sell_qty = fees = 0
        
        for trade in aggregated_trades:
            qty = float(trade['qty'])
            price = float(trade['price'])
            # Ensure timestamp is properly converted to datetime
            if isinstance(trade['time'], (int, float)):
                timestamp = datetime.fromtimestamp(float(trade['time']) / 1000)
            elif isinstance(trade['time'], str):
                timestamp = datetime.fromisoformat(trade['time'].replace('Z', '+00:00'))
            else:
                timestamp = trade['time']
            commission = float(trade['commission'])
            commission_asset = trade['commissionAsset']
            
            # Get market data and indicators
            market_data = get_market_data(client, symbol, timestamp)
            indicators = calculate_indicators(market_data)
            
            # Convert commission to USDT if needed
            if commission_asset != 'USDT':
                commission_value = commission * price
            else:
                commission_value = commission
            
            fees += commission_value
            
            if trade['isBuyer']:
                buy_queue.append((qty, price, timestamp, indicators))
            else:
                remaining_sell = qty
                while remaining_sell > 0 and buy_queue:
                    buy_qty, buy_price, buy_time, buy_indicators = buy_queue[0]
                    match_qty = min(buy_qty, remaining_sell)
                    
                    trade_details.append({
                        'buy_time': buy_time,
                        'sell_time': timestamp,
                        'buy_price': buy_price,
                        'sell_price': price,
                        'quantity': match_qty,
                        'buy_value': match_qty * buy_price,
                        'sell_value': match_qty * price,
                        'pnl': match_qty * (price - buy_price),
                        'hold_duration': (timestamp - buy_time).total_seconds() / 3600,
                        'commission': commission_value,
                        'buy_market_metrics': buy_indicators,
                        'sell_market_metrics': indicators
                    })
                    
                    if match_qty == buy_qty:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = (buy_qty - match_qty, buy_price, buy_time, buy_indicators)
                    
                    remaining_sell -= match_qty
                    matched_buy_value += match_qty * buy_price
                    matched_sell_value += match_qty * price
                    matched_buy_qty += match_qty
                    matched_sell_qty += match_qty
        
        return {
            'symbol': symbol,
            'buy_quantity': matched_buy_qty,
            'sell_quantity': matched_sell_qty,
            'buy_value': matched_buy_value,
            'sell_value': matched_sell_value,
            'realized_pnl': matched_sell_value - matched_buy_value,
            'fees': fees,
            'net_pnl': matched_sell_value - matched_buy_value - fees,
            'trade_details': trade_details
        }
        
    except Exception as e:
        logging.error(f"Error processing trades for {symbol}: {str(e)}")
        return None

def export_trade_results(results: List[Dict], start_time: datetime, 
                        end_time: datetime, output_file: str = 'trade_results.csv'):
    """Export enhanced trade results with market metrics"""
    if not results:
        logging.info("No trade data to export")
        return
    
    try:
        stats = calculate_trade_statistics(results)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Enhanced summary section
            writer.writerow(['Profitable Trades', stats['profitable_trades']])
            writer.writerow(['Unprofitable Trades', stats['unprofitable_trades']])
            writer.writerow(['Success Rate', f"{stats['success_rate']:.2f}%"])
            writer.writerow(['Gross Profit', f"{stats['total_profit']:.8f}"])
            writer.writerow(['Gross Loss', f"{stats['total_loss']:.8f}"])
            writer.writerow(['Total Fees', f"{stats['total_fees']:.8f}"])
            writer.writerow(['Net Profit', f"{stats['net_profit']:.8f}"])
            writer.writerow(['Average Hold Duration (H)', f"{stats['avg_hold_duration']:.2f}"])
            writer.writerow([])
            
            # Enhanced trade details with market metrics
            writer.writerow([
                'Symbol',
                'Buy Time',
                'Sell Time',
                'Buy Price',
                'Sell Price',
                'Quantity',
                'Buy Value',
                'Sell Value',
                'Gross P&L',
                'Fees',
                'Net P&L',
                'Hold Duration (H)',
                # Buy market metrics
                'Buy RSI',
                'Buy MACD',
                'Buy MACD Signal',
                'Buy MACD Hist',
                'Buy VWAP',
                'Buy BB Upper',
                'Buy BB Middle',
                'Buy BB Lower',
                'Buy MA7',
                'Buy MA25',
                'Buy MA50',
                'Buy MA100',
                'Buy ATR',
                'Buy ADX',
                'Buy OBV',
                'Buy Volume 1H',
                'Buy Volume 4H',
                'Buy Volume 24H',
                # Sell market metrics
                'Sell RSI',
                'Sell MACD',
                'Sell MACD Signal',
                'Sell MACD Hist',
                'Sell VWAP',
                'Sell BB Upper',
                'Sell BB Middle',
                'Sell BB Lower',
                'Sell MA7',
                'Sell MA25',
                'Sell MA50',
                'Sell MA100',
                'Sell ATR',
                'Sell ADX',
                'Sell OBV',
                'Sell Volume 1H',
                'Sell Volume 4H',
                'Sell Volume 24H'
            ])
            
            for result in results:
                for trade in result['trade_details']:
                    metrics_buy = trade['buy_market_metrics']
                    metrics_sell = trade['sell_market_metrics']
                    
                    row = [
                        result['symbol'],
                        trade['buy_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        trade['sell_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        f"{trade['buy_price']:.8f}",
                        f"{trade['sell_price']:.8f}",
                        f"{trade['quantity']:.8f}",
                        f"{trade['buy_value']:.8f}",
                        f"{trade['sell_value']:.8f}",
                        f"{trade['pnl']:.8f}",
                        f"{trade['commission']:.8f}",
                        f"{trade['pnl'] - trade['commission']:.8f}",
                        f"{trade['hold_duration']:.2f}"
                    ]
                    
                    # Add buy market metrics
                    for metric in [
                        'rsi', 'macd', 'macd_signal', 'macd_hist', 'vwap',
                        'bb_upper', 'bb_middle', 'bb_lower',
                        'ma7', 'ma25', 'ma50', 'ma100',
                        'atr', 'adx', 'obv',
                        'volume_1h', 'volume_4h', 'volume_24h'
                    ]:
                        row.append(f"{metrics_buy.get(metric, ''):.8f}")
                    
                    # Add sell market metrics
                    for metric in [
                        'rsi', 'macd', 'macd_signal', 'macd_hist', 'vwap',
                        'bb_upper', 'bb_middle', 'bb_lower',
                        'ma7', 'ma25', 'ma50', 'ma100',
                        'atr', 'adx', 'obv',
                        'volume_1h', 'volume_4h', 'volume_24h'
                    ]:
                        row.append(f"{metrics_sell.get(metric, ''):.8f}")
                    
                    writer.writerow(row)
        
        logging.info(f"\nDetailed trade results exported to {output_file}")
        
        # Print summary to console
        print("\n=== Trading Statistics ===")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Profitable Trades: {stats['profitable_trades']}")
        print(f"Unprofitable Trades: {stats['unprofitable_trades']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"Average Hold Duration: {stats['avg_hold_duration']:.2f} hours")
        print(f"Gross Profit: {stats['total_profit']:.8f} USDT")
        print(f"Gross Loss: {stats['total_loss']:.8f} USDT")
        print(f"Total Fees: {stats['total_fees']:.8f} USDT")
        print(f"Net Profit: {stats['net_profit']:.8f} USDT")
    
    except Exception as e:
        logging.error(f"Error exporting results: {str(e)}")

def main():
    """Main execution function"""
    try:
        setup_logging()
        logging.info("\nCrypto Trading Analysis")
        logging.info("=" * 80)
        
        # Initialize Binance client
        logging.info("Connecting to Binance...")
        client = load_binance_client()
        logging.info("✓ Connected to Binance successfully")
        
        # Define default symbols you want to analyze
        default_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'BCHUSDT', 'LTCUSDT'
        ]
        
        # Try to load symbols from checkpoint, falling back to defaults if needed
        logging.info("\nLoading trading symbols...")
        symbols = load_symbol_data(
            checkpoint_file='crypto_data_checkpoint.json',
            default_symbols=default_symbols
        )
        logging.info(f"✓ Loaded {len(symbols)} symbols for analysis")
        
        # Set time period
        now = datetime.now()
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        results = []
        
        # Initialize progress bar
        with tqdm(total=len(symbols), desc="Processing symbols", unit="symbol") as pbar:
            for symbol in symbols:
                try:
                    pbar.set_description(f"Processing {symbol}")
                    trades = get_trades_for_period(client, symbol, start_time, now)
                    if trades:
                        result = process_symbol_trades(client, trades)
                        if result:
                            results.append(result)
                            pbar.set_postfix(trades=len(trades))
                    time.sleep(0.1)  # Rate limiting
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
                    pbar.update(1)
                    continue
        
        if results:
            export_trade_results(results, start_time, now)
        else:
            logging.info("No trades found in the analysis period")
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()