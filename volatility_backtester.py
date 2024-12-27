import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import logging
import json
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_results.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_time: datetime
    target_price: float
    initial_amount: float  # Amount invested in this position
    status: str
    exit_price: float = 0
    exit_time: datetime = None
    profit_percent: float = 0
    days_in_trade: float = 0

class StrategyBacktester:
    def __init__(self, csv_file: str, max_positions: int = 10,
                 min_daily_volume_usdt: float = 10000000,
                 volatility_lookback: int = 5,
                 initial_capital: float = 1500):
        self.max_positions = max_positions
        self.min_daily_volume_usdt = min_daily_volume_usdt
        self.volatility_lookback = volatility_lookback
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = initial_capital / max_positions
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.daily_results = defaultdict(lambda: {"total_trades": 0, "success_rate": 0})
        self.historical_data = self._load_csv_data(csv_file)
        self.selected_symbols = set()
        
    def _calculate_metrics(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate volatility and volume metrics for a symbol"""
        df['returns'] = df['Close'].pct_change()
        volatility = df['returns'].std() * 100
        avg_volume = df['Quote_Volume'].mean()
        return volatility, avg_volume

    def _select_trading_symbols(self, date: datetime) -> List[str]:
        """Select top symbols based on volatility and volume"""
        metrics = []
        
        for symbol in set([s for d in self.historical_data.values() for s in d.keys()]):
            symbol_data = []
            current_date = date
            days_back = 0
            
            while days_back < self.volatility_lookback:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str in self.historical_data and symbol in self.historical_data[date_str]:
                    symbol_data.append(self.historical_data[date_str][symbol])
                current_date -= timedelta(days=1)
                days_back += 1
            
            if len(symbol_data) < self.volatility_lookback:
                continue
                
            df = pd.DataFrame(symbol_data)
            volatility, volume = self._calculate_metrics(df)
            
            if volume >= self.min_daily_volume_usdt:
                metrics.append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'volume': volume
                })
        
        if metrics:
            df_metrics = pd.DataFrame(metrics)
            df_metrics['volatility_rank'] = df_metrics['volatility'].rank(ascending=False)
            df_metrics['volume_rank'] = df_metrics['volume'].rank(ascending=False)
            df_metrics['combined_rank'] = df_metrics['volatility_rank'] + df_metrics['volume_rank']
            selected = df_metrics.nsmallest(self.max_positions, 'combined_rank')['symbol'].tolist()
            return selected
            
        return []

    def _load_csv_data(self, csv_file: str) -> Dict:
        """Load and process CSV data into required format"""
        logging.info("Loading CSV data...")
        
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        historical_data = {}
        
        for date in df['Date'].dt.date.unique():
            date_df = df[df['Date'].dt.date == date]
            date_data = {}
            
            for symbol in date_df['Symbol'].unique():
                symbol_df = date_df[date_df['Symbol'] == symbol]
                
                if len(symbol_df) == 0:
                    continue
                
                entry_price = float(symbol_df['Close'].iloc[0])
                
                date_data[symbol] = {
                    'entry_price': entry_price,
                    'Close': float(symbol_df['Close'].iloc[-1]),
                    'High': float(symbol_df['High'].max()),
                    'Low': float(symbol_df['Low'].min()),
                    'Quote_Volume': float(symbol_df['Quote_Volume'].sum()),
                    'price_data': {
                        '24h': {
                            'max': float(symbol_df['High'].max()),
                            'min': float(symbol_df['Low'].min())
                        },
                        '48h': {
                            'max': float(symbol_df['High'].max()),
                            'min': float(symbol_df['Low'].min())
                        }
                    }
                }
            
            historical_data[date.strftime('%Y-%m-%d')] = date_data
            
        logging.info(f"Loaded data for {len(historical_data)} days")
        return historical_data

    def backtest_strategy(self):
        """Run backtest using loaded CSV data"""
        for date_str, date_data in self.historical_data.items():
            current_date = datetime.strptime(date_str, '%Y-%m-%d')
            logging.info(f"\nProcessing date: {date_str}")
            
            self.selected_symbols = set(self._select_trading_symbols(current_date))
            logging.info(f"Selected symbols for trading: {self.selected_symbols}")
            
            self._update_positions(current_date, date_data)
            
            available_slots = self.max_positions - len(self.active_positions)
            if available_slots > 0:
                self._open_positions(current_date, date_data, available_slots)
            
            self._calculate_daily_metrics(current_date)
        
        self._save_results()

    def _open_positions(self, current_date: datetime, date_data: Dict, available_slots: int):
        """Open new positions based on historical data"""
        opportunities = []
        current_position_size = self.current_capital / self.max_positions
        
        for symbol in self.selected_symbols:
            if symbol in date_data and len(opportunities) < available_slots:
                data = date_data[symbol]
                
                daily_range = (data['High'] - data['Low']) / data['Low'] * 100
                if daily_range >= 1.0:
                    entry_price = data['entry_price'] * 0.995  # -0.5%
                    target_price = entry_price * 1.015  # +1.5%
                    
                    opportunities.append(Position(
                        symbol=symbol,
                        entry_price=entry_price,
                        entry_time=current_date,
                        target_price=target_price,
                        initial_amount=current_position_size,
                        status='OPEN'
                    ))
        
        self.active_positions.extend(opportunities[:available_slots])

    def _update_positions(self, current_date: datetime, date_data: Dict):
        """Update active positions based on historical price data"""
        for position in self.active_positions[:]:
            symbol = position.symbol
            if symbol not in date_data:
                continue
            
            price_data = date_data[symbol]['price_data']
            
            if price_data['24h']['max'] >= position.target_price:
                self._close_position(position, current_date, position.target_price, 1.5)
            elif (current_date - position.entry_time).days >= 2:
                exit_price = price_data['48h']['max']
                profit_percent = ((exit_price / position.entry_price) - 1) * 100
                self._close_position(position, current_date, exit_price, profit_percent)
    
    def _close_position(self, position: Position, exit_time: datetime, exit_price: float, profit_percent: float):
        """Close a position and move it to closed positions"""
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.profit_percent = profit_percent
        position.days_in_trade = (exit_time - position.entry_time).total_seconds() / (24 * 3600)
        position.status = 'CLOSED'
        
        # Update capital (compound interest)
        profit = position.initial_amount * (profit_percent / 100)
        self.current_capital += profit
        
        self.closed_positions.append(position)
        self.active_positions.remove(position)
    
    def _calculate_daily_metrics(self, date: datetime):
        """Calculate success rates for the day"""
        day_positions = [p for p in self.closed_positions if p.exit_time.date() == date.date()]
        
        if not day_positions:
            return
        
        total_trades = len(day_positions)
        successful_trades = len([p for p in day_positions if p.profit_percent > 0])
        
        self.daily_results[date.date()] = {
            "total_trades": total_trades,
            "success_rate": (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        }
    
    def _save_results(self):
        """Save backtest results to file"""
        trade_durations = [p.days_in_trade for p in self.closed_positions]
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        min_duration = min(trade_durations) if trade_durations else 0
        max_duration = max(trade_durations) if trade_durations else 0
        
        profits = [p.profit_percent for p in self.closed_positions]
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_percent': ((self.current_capital / self.initial_capital) - 1) * 100,
            'total_trades': len(self.closed_positions),
            'average_trade_duration_days': avg_duration,
            'min_trade_duration_days': min_duration,
            'max_trade_duration_days': max_duration,
            'average_profit_per_trade': np.mean(profits) if profits else 0,
            'win_rate': len([p for p in self.closed_positions if p.profit_percent > 0]) / len(self.closed_positions) * 100 if self.closed_positions else 0,
            'trade_history': [{
                'symbol': p.symbol,
                'entry_time': str(p.entry_time),
                'exit_time': str(p.exit_time),
                'days_in_trade': p.days_in_trade,
                'entry_price': p.entry_price,
                'exit_price': p.exit_price,
                'profit_percent': p.profit_percent,
                'position_size': p.initial_amount
            } for p in self.closed_positions]
        }
        
        with open('strategy_results_2024.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("\nBacktest Complete!")
        logging.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logging.info(f"Final Capital: ${self.current_capital:,.2f}")
        logging.info(f"Total Return: {results['total_return_percent']:.2f}%")
        logging.info(f"Total Trades: {results['total_trades']}")
        logging.info(f"Win Rate: {results['win_rate']:.2f}%")
        logging.info(f"\nTrade Duration Statistics:")
        logging.info(f"Average Days in Trade: {avg_duration:.2f}")
        logging.info(f"Minimum Days in Trade: {min_duration:.2f}")
        logging.info(f"Maximum Days in Trade: {max_duration:.2f}")
        logging.info(f"\nPosition Size Information:")
        logging.info(f"Initial Position Size: ${self.initial_capital/self.max_positions:,.2f}")
        logging.info(f"Final Position Size: ${self.current_capital/self.max_positions:,.2f}")

if __name__ == "__main__":
    csv_file = os.path.join('binance_data', 'binance_all_symbols_2024.csv')
    
    if not os.path.exists(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        logging.info("Please ensure the data collection script has been run first")
        exit(1)
        
    backtester = StrategyBacktester(
        csv_file=csv_file,
        max_positions=10,
        min_daily_volume_usdt=10000000,  # 10M USDT minimum volume
        volatility_lookback=5,  # 5-day lookback for volatility
        initial_capital=1500  # Starting with $1500
    )
    
    backtester.backtest_strategy()