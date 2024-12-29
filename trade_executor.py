import logging
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import Config
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client
from services.trade_analyzer import TradeAnalyzer, TradeSignal

class TradeAllocator:
    """Handles trade allocation based on market conditions and signal strength"""
    
    def __init__(self, total_amount: float, min_trade: float = 15, max_trade: float = 100):
        self.total_amount = total_amount
        self.min_trade = min_trade
        self.max_trade = max_trade
        
    def calculate_score(self, signal: TradeSignal) -> float:
        """Calculate a score for trade allocation based on signal strength and conditions"""
        # Count met conditions
        conditions_met = sum(1 for cond in signal.market_conditions.values() if cond.met)
        total_conditions = len(signal.market_conditions)
        condition_score = conditions_met / total_conditions
        
        # Calculate volatility score (lower volatility = higher score)
        volatility = next((cond.value for name, cond in signal.market_conditions.items() 
                         if 'volatility' in name.lower()), 0.03)
        volatility_score = max(0, 1 - (volatility / 0.05))  # Normalize volatility
        
        # Combine scores with weights
        score = (
            0.4 * signal.prediction +      # Model prediction
            0.3 * condition_score +        # Market conditions
            0.3 * volatility_score         # Volatility impact
        )
        
        return score
        
    def allocate_amounts(self, signals: Dict[str, TradeSignal]) -> Dict[str, float]:
        """Allocate trading amounts based on signals and scores"""
        # Filter valid signals and calculate scores
        valid_trades: List[Tuple[str, TradeSignal, float]] = []
        
        for symbol, signal in signals.items():
            if signal and signal.signal_found and signal.prediction > 0.6:
                score = self.calculate_score(signal)
                valid_trades.append((symbol, signal, score))
                
        if not valid_trades:
            return {}
            
        # Sort by score
        valid_trades.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate allocations
        total_score = sum(score for _, _, score in valid_trades)
        allocations = {}
        remaining_amount = self.total_amount
        
        for symbol, signal, score in valid_trades:
            if remaining_amount <= 0:
                break
                
            # Calculate raw allocation based on score
            allocation = (score / total_score) * self.total_amount
            
            # Apply min/max constraints
            allocation = min(max(allocation, self.min_trade), self.max_trade)
            allocation = min(allocation, remaining_amount)
            
            if allocation >= self.min_trade:
                allocations[symbol] = allocation
                remaining_amount -= allocation
                
        return allocations

class TradeExecutor:
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading info for a symbol"""
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                raise ValueError(f"No symbol info found for {symbol}")
            return info
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def format_price(self, price: float, symbol_info: Dict) -> str:
        """Format price according to symbol's price filter"""
        price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', symbol_info['filters']))
        tick_size = float(price_filter['tickSize'])
        
        # Calculate price precision from tick size
        precision = len(str(tick_size).rstrip('0').split('.')[-1])
        formatted_price = f"{price:.{precision}f}"
        
        return formatted_price

    def calculate_quantity(self, symbol: str, entry_price: float, usdt_amount: float) -> Optional[float]:
        """Calculate quantity based on USDT amount and current price"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
                
            # Get the lot size filter
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
            min_qty = float(lot_size['minQty'])
            step_size = float(lot_size['stepSize'])
            
            # Calculate raw quantity
            quantity = usdt_amount / entry_price
            
            # Round down to valid step size
            decimal_places = len(str(step_size).rstrip('0').split('.')[-1])
            quantity = float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
            
            # Check minimum quantity
            if quantity < min_qty:
                self.logger.warning(f"{symbol}: Calculated quantity {quantity} is below minimum {min_qty}")
                return None
                
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity for {symbol}: {e}")
            return None
            
    def wait_for_order_fill(self, symbol: str, order_id: str, timeout: int = 60) -> bool:
        """Wait for an order to be filled"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            if order['status'] == 'FILLED':
                return True
            elif order['status'] in ['CANCELED', 'REJECTED', 'EXPIRED']:
                self.logger.error(f"Order failed with status: {order['status']}")
                return False
            time.sleep(2)  # Wait 2 seconds before checking again
        return False

    def execute_trade(self, symbol: str, signal: TradeSignal, usdt_amount: float) -> bool:
        """Execute a trade with limit entry, TP, and SL"""
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False

            # Calculate quantity
            quantity = self.calculate_quantity(symbol, signal.entry_price, usdt_amount)
            if not quantity:
                return False

            # Format prices according to symbol's price filter
            entry_price = self.format_price(signal.entry_price, symbol_info)
            target_price = self.format_price(signal.target_price, symbol_info)
            stop_price = self.format_price(signal.stop_price, symbol_info)

            # Print formatted prices for verification
            self.logger.info(f"Formatted prices for {symbol}:")
            self.logger.info(f"  Entry: {entry_price}")
            self.logger.info(f"  Target: {target_price}")
            self.logger.info(f"  Stop: {stop_price}")
                
            # Place limit entry order
            entry_order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=entry_price
            )
            
            self.logger.info(f"Placed entry order for {symbol}:")
            self.logger.info(f"  Amount: {usdt_amount} USDT")
            self.logger.info(f"  Quantity: {quantity}")
            self.logger.info(f"  Entry Price: {entry_price}")

            # Wait for entry order to be filled
            if not self.wait_for_order_fill(symbol, entry_order['orderId']):
                self.logger.error(f"Entry order for {symbol} was not filled within timeout")
                return False

            # Place OCO orders after entry is filled
            params = {
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": target_price,  # Take profit price
                "stopPrice": stop_price,
                "stopLimitPrice": stop_price,
                "stopLimitTimeInForce": "GTC",
                "listClientOrderId": f"oco_{symbol}_{int(time.time())}"
            }

            # Create OCO order using low-level API call
            oco_order = self.client._request_margin_api('post', 'order/oco', True, data=params)
            
            self.logger.info(f"Placed OCO order for {symbol}:")
            self.logger.info(f"  Take Profit: {target_price}")
            self.logger.info(f"  Stop Loss: {stop_price}")
            
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error for {symbol}: {e}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response: {e.response.text}")
            self.logger.error(f"Attempted values - Quantity: {quantity}, Entry: {entry_price}, Target: {target_price}, Stop: {stop_price}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False

def execute_trades():
    """Main function to find and execute trades"""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    config = Config()
    
    if not Path('.env').exists():
        logger.error("❌ .env file not found")
        return
        
    try:
        # Get total trade amount from user
        total_amount = float(input("Enter the total amount of USDT to trade: "))
        if total_amount <= 0:
            raise ValueError("Trade amount must be positive")
            
        # Initialize
        client = setup_binance_client()
        analyzer = TradeAnalyzer(client, config)
        executor = TradeExecutor(client)
        allocator = TradeAllocator(total_amount)
        
        print("\n=== Starting Trade Execution ===")
        print(f"Total Trade Amount: {total_amount} USDT")
        
        # Check account balance
        account = client.get_account()
        usdt_balance = float(next(asset['free'] for asset in account['balances'] if asset['asset'] == 'USDT'))
        
        if usdt_balance < total_amount:
            logger.error(f"Insufficient USDT balance. Available: {usdt_balance}")
            return
            
        print(f"Available USDT: {usdt_balance}")
        
        # Initialize traders
        if not analyzer.initialize_traders():
            logger.error("Failed to initialize traders")
            return
            
        # Generate signals
        signals = analyzer.generate_trading_signals()
        
        # Allocate amounts
        allocations = allocator.allocate_amounts(signals)
        
        if not allocations:
            print("No valid trading signals found")
            return
            
        # Print allocation plan
        print("\n=== Trade Allocation Plan ===")
        for symbol, amount in allocations.items():
            signal = signals[symbol]
            print(f"\n{symbol}:")
            print(f"  Allocated Amount: {amount:.2f} USDT")
            print(f"  Signal Strength: {signal.prediction:.2%}")
            print(f"  Risk/Reward: {(signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_price):.2f}")
            
        # Confirm with user
        confirm = input("\nProceed with trades? (yes/no): ").lower()
        if confirm != 'yes':
            print("Trading cancelled")
            return
            
        # Execute trades
        executed_trades = 0
        total_invested = 0
        
        for symbol, amount in allocations.items():
            signal = signals[symbol]
            print(f"\nExecuting trade for {symbol}...")
            
            if executor.execute_trade(symbol, signal, amount):
                executed_trades += 1
                total_invested += amount
                print(f"✅ Successfully executed trade for {symbol}")
            else:
                print(f"❌ Failed to execute trade for {symbol}")
                
            time.sleep(1)  # Delay between trades
            
        print(f"\n=== Trade Execution Summary ===")
        print(f"Total signals analyzed: {len(signals)}")
        print(f"Trades executed: {executed_trades}")
        print(f"Total amount invested: {total_invested:.2f} USDT")
        print(f"Remaining allocation: {total_amount - total_invested:.2f} USDT")
        
    except Exception as e:
        logger.error(f"Error during trade execution: {e}")
        raise

if __name__ == "__main__":
    execute_trades()