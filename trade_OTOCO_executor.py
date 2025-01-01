import logging
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from binance.client import Client
import hmac
import hashlib
import requests
from binance.exceptions import BinanceAPIException
from config import Config
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client
from services.trade_analyzer import TradeAnalyzer, TradeSignal

class TradeAllocator:
    """Enhanced trade allocation based on market conditions and signal strength"""
    
    def __init__(self, total_amount: float, min_trade: float = 15, max_trade: float = 100):
        self.total_amount = total_amount
        self.min_trade = min_trade
        self.max_trade = max_trade
        self.logger = logging.getLogger(__name__)
        
    def calculate_score(self, signal: TradeSignal) -> float:
        """Calculate enhanced allocation score based on technical patterns"""
        try:
            # Volume score (higher weight due to strong correlation with success)
            volume_condition = next(c for name, c in signal.market_conditions.items() 
                                  if 'volume' in name.lower())
            volume_score = volume_condition.value / volume_condition.threshold

            # Volatility score (inverse relationship - lower is better)
            volatility_condition = next(c for name, c in signal.market_conditions.items() 
                                      if 'volatility' in name.lower())
            volatility_score = max(0, 1 - (volatility_condition.value / volatility_condition.threshold))

            # RSI score (highest near 50)
            rsi_condition = next(c for name, c in signal.market_conditions.items() 
                               if 'rsi' in name.lower())
            rsi_center = 50
            rsi_score = 1 - abs(rsi_condition.value - rsi_center) / 50

            # Technical conditions combined score
            conditions_met = sum(1 for cond in signal.market_conditions.values() if cond.met)
            condition_score = conditions_met / len(signal.market_conditions)

            # Risk/reward consideration
            risk_reward_score = min(1.0, signal.risk_reward_ratio / 3.0)

            # Weighted final score
            score = (
                0.30 * signal.prediction +      # Model prediction
                0.25 * volume_score +           # Volume impact
                0.15 * condition_score +        # Overall conditions
                0.15 * risk_reward_score +      # Risk/reward ratio
                0.10 * volatility_score +       # Volatility impact
                0.05 * rsi_score               # RSI optimization
            )
            
            return score

        except Exception as e:
            self.logger.error(f"Error calculating score: {str(e)}")
            return 0.0
        
    def allocate_amounts(self, signals: Dict[str, TradeSignal]) -> Dict[str, float]:
        """Allocate trading amounts based on enhanced scoring"""
        try:
            # Filter valid signals and calculate scores
            valid_trades: List[Tuple[str, TradeSignal, float]] = []
            
            for symbol, signal in signals.items():
                if signal and signal.signal_found and signal.prediction > 0.8:
                    score = self.calculate_score(signal)
                    if score > 0.7:  # Minimum score threshold
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
                if remaining_amount <= self.min_trade:
                    break
                    
                # Calculate allocation based on score
                base_allocation = (score / total_score) * self.total_amount
                
                # Adjust based on risk/reward
                risk_factor = min(1.2, max(0.8, signal.risk_reward_ratio / 2))
                adjusted_allocation = base_allocation * risk_factor
                
                # Apply constraints
                final_allocation = min(
                    max(adjusted_allocation, self.min_trade),
                    min(self.max_trade, remaining_amount)
                )
                
                if final_allocation >= self.min_trade:
                    allocations[symbol] = final_allocation
                    remaining_amount -= final_allocation
                    
            return allocations

        except Exception as e:
            self.logger.error(f"Error allocating amounts: {str(e)}")
            return {}

class TradeExecutor:
    """Enhanced trade execution with improved risk management"""
    
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading info with error handling"""
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
        try:
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', 
                                     symbol_info['filters']))
            tick_size = float(price_filter['tickSize'])
            
            precision = len(str(tick_size).rstrip('0').split('.')[-1])
            formatted_price = f"{price:.{precision}f}"
            
            return formatted_price
        except Exception as e:
            self.logger.error(f"Error formatting price: {e}")
            return None

    def calculate_quantity(self, symbol: str, entry_price: float, usdt_amount: float) -> Optional[float]:
        """Calculate quantity with enhanced precision handling and lot size compliance"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
                
            # Get lot size filter
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                symbol_info['filters']))
            min_qty = float(lot_size['minQty'])
            max_qty = float(lot_size.get('maxQty', float('inf')))
            step_size = float(lot_size['stepSize'])
            
            # Calculate initial quantity
            quantity = usdt_amount / entry_price
            
            # Adjust quantity to meet step size requirements
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            quantity = round(quantity / step_size) * step_size
            
            # Ensure quantity meets minimum requirements
            if quantity < min_qty:
                self.logger.warning(
                    f"{symbol}: Calculated quantity {quantity} below minimum {min_qty}. "
                    f"Adjusting to minimum lot size."
                )
                quantity = min_qty
            
            # Ensure quantity does not exceed maximum (if specified)
            if quantity > max_qty:
                self.logger.warning(
                    f"{symbol}: Calculated quantity {quantity} exceeds maximum {max_qty}. "
                    f"Adjusting to maximum lot size."
                )
                quantity = max_qty
            
            # Round to appropriate precision
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal(str(step_size)), 
                rounding=ROUND_DOWN
            ))
            
            self.logger.info(f"{symbol}: Calculated quantity: {quantity}")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity for {symbol}: {e}")
            return None

    def execute_trade(self, symbol: str, signal: TradeSignal, usdt_amount: float) -> bool:
        """Execute trade with enhanced OTOCO order and dynamic price levels"""
        try:
            # Get market conditions
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False

            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            market_price = float(ticker['price'])

            # Calculate quantity
            quantity = self.calculate_quantity(symbol, signal.entry_price, usdt_amount)
            if not quantity:
                self.logger.error(f"Failed to calculate valid quantity for {symbol}")
                return False

            # Log detailed quantity calculation details
            self.logger.info(f"{symbol} Quantity Calculation:")
            self.logger.info(f"  USDT Amount: {usdt_amount}")
            self.logger.info(f"  Entry Price: {signal.entry_price}")
            self.logger.info(f"  Calculated Quantity: {quantity}")

            # Format prices
            price_filter = next(f for f in symbol_info['filters'] 
                              if f['filterType'] == 'PRICE_FILTER')
            decimals = len(str(float(price_filter['tickSize'])).rstrip('0').split('.')[-1])

            # Dynamic price levels based on volatility
            entry_price = round(market_price * 0.99, decimals)  # Slight discount to market
            target_price = round(signal.target_price, decimals)
            stop_price = round(signal.stop_price, decimals)

            # Log trade parameters
            self._log_trade_parameters(symbol, market_price, entry_price, 
                                     target_price, stop_price, quantity)

            # Build OTOCO order
            otoco_params = self._build_otoco_params(
                symbol, entry_price, target_price, stop_price, quantity
            )

            # Generate signature
            query_string = self._generate_signature(otoco_params)

            # Execute order
            success = self._send_otoco_order(symbol, query_string)
            
            return success

        except BinanceAPIException as e:
            # Specific Binance API error handling
            self.logger.error(f"Binance API Error for {symbol}: {e.status_code} - {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error executing trade for {symbol}: {e}")
            return False

    def _log_trade_parameters(self, symbol: str, market_price: float, entry_price: float,
                            target_price: float, stop_price: float, quantity: float):
        """Log detailed trade parameters"""
        self.logger.info(f"Trade parameters for {symbol}:")
        self.logger.info(f"  Market price: {market_price}")
        self.logger.info(f"  Entry: {entry_price}")
        self.logger.info(f"  Target: {target_price}")
        self.logger.info(f"  Stop: {stop_price}")
        self.logger.info(f"  Quantity: {quantity}")

    def _build_otoco_params(self, symbol: str, entry_price: float, target_price: float,
                           stop_price: float, quantity: float) -> Dict:
        """Build OTOCO order parameters"""
        return {
            "symbol": symbol,
            "timestamp": str(int(time.time() * 1000)),
            
            # Entry Order
            "workingType": "LIMIT",
            "workingSide": "BUY",
            "workingTimeInForce": "GTC",
            "workingPrice": str(entry_price),
            "workingQuantity": str(quantity),
            
            # Exit Orders
            "pendingSide": "SELL",
            "pendingQuantity": str(quantity),
            "pendingAboveType": "LIMIT_MAKER",
            "pendingAbovePrice": str(target_price),
            "pendingBelowType": "STOP_LOSS",
            "pendingBelowStopPrice": str(stop_price),
            
            "recvWindow": "5000"
        }

    def _generate_signature(self, params: Dict) -> str:
        """Generate API request signature"""
        sorted_params = dict(sorted(params.items()))
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_params.items()])
        
        signature = hmac.new(
            bytes(self.client.API_SECRET, 'utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return f"{query_string}&signature={signature}"

    def _send_otoco_order(self, symbol: str, query_string: str) -> bool:
        """Send OTOCO order to exchange"""
        try:
            endpoint = 'https://api.binance.com/api/v3/orderList/otoco'
            headers = {
                'X-MBX-APIKEY': self.client.API_KEY,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            self.logger.info(f"Sending OTOCO request for {symbol}")
            response = requests.post(f"{endpoint}?{query_string}", headers=headers)

            if response.status_code != 200:
                self.logger.error(f"API Error: {response.status_code} - {response.text}")
                return False

            self._log_order_response(response.json())
            return True

        except Exception as e:
            self.logger.error(f"Error sending OTOCO order: {e}")
            return False

    def _log_order_response(self, response_data: Dict):
        """Log detailed order response"""
        self.logger.info(f"OTOCO order response:")
        self.logger.info(f"  Order List ID: {response_data.get('orderListId')}")
        self.logger.info(f"  Status: {response_data.get('listStatusType')}")
        
        for report in response_data.get('orderReports', []):
            self.logger.info(f"\n  {report.get('type', '')} Order:")
            self.logger.info(f"    Order ID: {report.get('orderId')}")
            self.logger.info(f"    Side: {report.get('side')}")
            self.logger.info(f"    Price: {report.get('price')}")
            if 'stopPrice' in report:
                self.logger.info(f"    Stop Price: {report.get('stopPrice')}")
            self.logger.info(f"    Status: {report.get('status')}")

def execute_trades():
    """Main function to execute trades with enhanced validation"""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    config = Config()
    
    if not Path('.env').exists():
        logger.error("❌ .env file not found")
        return
        
    try:
        # Get trade amount
        total_amount = float(input("Enter total USDT amount to trade: "))
        if total_amount <= 0:
            raise ValueError("Trade amount must be positive")
            
        # Initialize components
        client = setup_binance_client()
        analyzer = TradeAnalyzer(client, config)
        executor = TradeExecutor(client)
        allocator = TradeAllocator(total_amount)
        
        logger.info("\n=== Starting Enhanced Trade Execution ===")
        logger.info(f"Total Trade Amount: {total_amount} USDT")
        
        # Validate balance
        account = client.get_account()
        usdt_balance = float(next(
            asset['free'] for asset in account['balances'] 
            if asset['asset'] == 'USDT'
        ))
        
        if usdt_balance < total_amount:
            logger.error(f"Insufficient USDT balance. Available: {usdt_balance}")
            return
            
        logger.info(f"Available USDT: {usdt_balance}")
        
        # Initialize system
        if not analyzer.initialize_traders():
            logger.error("Failed to initialize traders")
            return
            
        # Generate and analyze signals
        signals = analyzer.generate_trading_signals()
        if not signals:
            logger.warning("No valid trading signals found")
            return
            
        # Allocate trading amounts
        allocations = allocator.allocate_amounts(signals)
        if not allocations:
            logger.warning("No valid allocations generated")
            return
            
        # Display allocation plan
        logger.info("\n=== Trade Allocation Plan ===")
        for symbol, amount in allocations.items():
            signal = signals[symbol]
            logger.info(f"\n{symbol}:")
            logger.info(f"  Allocated Amount: {amount:.2f} USDT")
            logger.info(f"  Signal Strength: {signal.prediction:.2%}")
            logger.info(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
            logger.info(f"  Potential Profit: {signal.potential_profit:.2f}%")
            logger.info(f"  Maximum Risk: {signal.maximum_risk:.2f}%")
            logger.info(f"  Market Conditions Met: {sum(1 for c in signal.market_conditions.values() if c.met)}/{len(signal.market_conditions)}")
            
        # Get user confirmation
        confirm = input("\nProceed with trades? (yes/no): ").lower()
        if confirm != 'yes':
            logger.info("Trading cancelled by user")
            return
            
        # Execute trades with enhanced monitoring
        executed_trades = 0
        total_invested = 0
        failed_trades = []
        
        for symbol, amount in allocations.items():
            signal = signals[symbol]
            logger.info(f"\nExecuting trade for {symbol}...")
            
            # Pre-trade validation
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            price_drift = abs(current_price - signal.current_price) / signal.current_price
            
            if price_drift > 0.005:  # More than 0.5% price drift
                logger.warning(f"Price drifted {price_drift:.2%} from signal generation. Skipping {symbol}")
                failed_trades.append((symbol, "Price drift too high"))
                continue
                
            # Execute trade with retry mechanism
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count} for {symbol}")
                    time.sleep(1)  # Wait before retry
                    
                success = executor.execute_trade(symbol, signal, amount)
                retry_count += 1
                
            if success:
                executed_trades += 1
                total_invested += amount
                logger.info(f"✅ Successfully executed trade for {symbol}")
            else:
                failed_trades.append((symbol, "Execution failed after retries"))
                logger.error(f"❌ Failed to execute trade for {symbol}")
                
            time.sleep(1)  # Delay between trades
            
        # Print detailed execution summary
        logger.info(f"\n=== Trade Execution Summary ===")
        logger.info(f"Total signals analyzed: {len(signals)}")
        logger.info(f"Trades attempted: {len(allocations)}")
        logger.info(f"Trades executed successfully: {executed_trades}")
        logger.info(f"Trades failed: {len(failed_trades)}")
        logger.info(f"Total amount invested: {total_invested:.2f} USDT")
        logger.info(f"Remaining allocation: {total_amount - total_invested:.2f} USDT")
        
        if failed_trades:
            logger.info("\nFailed Trades Details:")
            for symbol, reason in failed_trades:
                logger.info(f"  {symbol}: {reason}")
        
    except Exception as e:
        logger.error(f"Critical error during trade execution: {e}")
        raise

if __name__ == "__main__":
    execute_trades()