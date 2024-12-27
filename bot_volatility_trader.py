from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volatility_trader.log'),
        logging.StreamHandler()
    ]
)

class SimpleVolatilityTrader:
    def __init__(self, 
                 min_volume_usdt: float = 1000000,
                 lookback_days: int = 7,
                 position_size: float = 150):
        self.client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
        self.min_volume_usdt = min_volume_usdt
        self.lookback_days = lookback_days
        self.position_size = position_size
        logging.info(f"Initialized trader with position size: ${position_size}, min volume: ${min_volume_usdt}")
        
    def get_symbol_precision(self, symbol: str) -> Dict:
        """Get symbol's price and quantity precision"""
        info = self.client.get_symbol_info(symbol)
        price_precision = None
        qty_precision = None
        
        for filter in info['filters']:
            if filter['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter['tickSize'])
                price_precision = len(str(tick_size).rstrip('0').split('.')[1])
            elif filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                qty_precision = len(str(step_size).rstrip('0').split('.')[1])
                
        return {
            'price_precision': price_precision,
            'qty_precision': qty_precision
        }

    def place_orders(self, symbol: str, setup: Dict) -> Dict:
        """Place both entry and take profit orders"""
        try:
            precision = self.get_symbol_precision(symbol)
            quantity = round(setup['quantity'], precision['qty_precision'])
            entry_price = round(setup['entry_price'], precision['price_precision'])
            target_price = round(setup['target_price'], precision['price_precision'])
            
            logging.info(f"Attempting to place orders for {symbol}")
            logging.info(f"Entry: {quantity} {symbol} @ {entry_price} USDT")
            
            # Place limit buy order
            entry_order = self.client.create_order(
                symbol=symbol,
                side='BUY',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(entry_price)
            )
            
            logging.info(f"Entry order placed successfully: {entry_order['orderId']}")
            
            # Place take profit sell order
            take_profit_order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(target_price)
            )
            
            logging.info(f"Take profit order placed successfully: {take_profit_order['orderId']}")
            
            return {
                'entry_order_id': entry_order['orderId'],
                'tp_order_id': take_profit_order['orderId'],
                'status': 'success'
            }
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error placing orders: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def monitor_orders(self, symbol: str, entry_order_id: str, tp_order_id: str):
        """Monitor the status of placed orders"""
        try:
            entry_order = self.client.get_order(symbol=symbol, orderId=entry_order_id)
            tp_order = self.client.get_order(symbol=symbol, orderId=tp_order_id)
            
            logging.info(f"Entry order status: {entry_order['status']}")
            logging.info(f"Take profit order status: {tp_order['status']}")
            
            return {
                'entry_status': entry_order['status'],
                'tp_status': tp_order['status']
            }
            
        except BinanceAPIException as e:
            logging.error(f"Error monitoring orders: {str(e)}")
            return None

    def find_volatile_pairs(self) -> pd.DataFrame:
        """Find crypto pairs with high volatility and volume"""
        logging.info("Starting volatile pair search...")
        exchange_info = self.client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] 
                  if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        
        logging.info(f"Found {len(symbols)} USDT trading pairs")
        opportunities = []
        processed = 0
        
        for symbol in symbols:
            try:
                processed += 1
                if processed % 10 == 0:
                    logging.info(f"Processed {processed}/{len(symbols)} pairs")
                
                klines = self.client.get_historical_klines(
                    symbol,
                    Client.KLINE_INTERVAL_1HOUR,
                    str(datetime.now() - timedelta(days=self.lookback_days))
                )
                
                if len(klines) < 24:
                    continue
                    
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignored'
                ])
                
                df['close'] = pd.to_numeric(df['close'])
                df['quote_volume'] = pd.to_numeric(df['quote_volume'])
                
                current_price = float(df['close'].iloc[-1])
                avg_daily_volume = float(df['quote_volume'].mean() * 24)
                
                if avg_daily_volume < self.min_volume_usdt:
                    continue
                
                df['returns'] = df['close'].pct_change()
                hourly_volatility = df['returns'].std() * 100
                price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100
                
                price_direction = 'DOWN' if price_change_24h < -1 else 'UP' if price_change_24h > 1 else 'SIDEWAYS'
                
                if hourly_volatility > 0.5:  # Only log interesting opportunities
                    logging.info(f"Found volatile pair: {symbol} - Volatility: {hourly_volatility:.2f}% - 24h Change: {price_change_24h:.2f}%")
                
                opportunities.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'hourly_volatility': round(hourly_volatility, 2),
                    'daily_volume_usdt': round(avg_daily_volume, 2),
                    'price_change_24h': round(price_change_24h, 2),
                    'price_direction': price_direction,
                    'entry_price': round(current_price * 0.995, 8),
                    'target_price': round(current_price * 1.025, 8)
                })
                
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        df_opportunities = pd.DataFrame(opportunities)
        if not df_opportunities.empty:
            df_opportunities = df_opportunities[
                (df_opportunities['hourly_volatility'] > 0.5) &
                (df_opportunities['daily_volume_usdt'] > self.min_volume_usdt)
            ].sort_values('hourly_volatility', ascending=False)
            
            logging.info(f"Found {len(df_opportunities)} opportunities matching criteria")
        else:
            logging.info("No opportunities found matching criteria")
        
        return df_opportunities

    def get_trade_setup(self, symbol: str) -> Dict:
        """Get specific trade setup for a symbol"""
        try:
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            precision = self.get_symbol_precision(symbol)
            
            quantity = round(self.position_size / current_price, precision['qty_precision'])
            entry_price = round(current_price * 0.995, precision['price_precision'])
            target_price = round(current_price * 1.025, precision['price_precision'])
            
            setup = {
                'symbol': symbol,
                'position_size_usdt': self.position_size,
                'entry_price': entry_price,
                'target_price': target_price,
                'quantity': quantity,
                'current_price': current_price
            }
            
            logging.info(f"Trade setup for {symbol}:")
            for key, value in setup.items():
                logging.info(f"{key}: {value}")
                
            return setup
            
        except Exception as e:
            logging.error(f"Error getting trade setup for {symbol}: {str(e)}")
            return None

def run_trader():
    try:
        logging.info("Starting volatility trader...")
        trader = SimpleVolatilityTrader(min_volume_usdt=1000000, position_size=150)
        opportunities = trader.find_volatile_pairs()
        
        if not opportunities.empty:
            logging.info("\nTop 5 Volatile Pairs:")
            top_5 = opportunities.head()
            for _, row in top_5.iterrows():
                logging.info(f"{row['symbol']}: {row['hourly_volatility']}% volatility, "
                           f"${row['daily_volume_usdt']:,.2f} volume, {row['price_direction']} trend")
            
            # Get trade setup for top opportunity
            top_symbol = opportunities.iloc[0]['symbol']
            trade_setup = trader.get_trade_setup(top_symbol)
            
            if trade_setup:
                # Place orders
                orders = trader.place_orders(top_symbol, trade_setup)
                
                if orders['status'] == 'success':
                    logging.info("Orders placed successfully")
                    
                    # Monitor orders for 1 hour
                    for _ in range(12):  # Check every 5 minutes for 1 hour
                        status = trader.monitor_orders(
                            top_symbol, 
                            orders['entry_order_id'], 
                            orders['tp_order_id']
                        )
                        if status:
                            if status['entry_status'] == 'FILLED' and status['tp_status'] == 'FILLED':
                                logging.info("Both orders filled successfully!")
                                break
                        time.sleep(300)  # Wait 5 minutes between checks
                        
    except Exception as e:
        logging.error(f"Error in main trading loop: {str(e)}")

if __name__ == "__main__":
    run_trader()