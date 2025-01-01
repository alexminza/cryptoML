import pandas as pd
from datetime import datetime, timedelta
import os
from binance.client import Client
from dotenv import load_dotenv
import time
import json
import csv

def load_binance_client():
    """Initialize Binance client with credentials from .env"""
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file")
    
    return Client(api_key, api_secret)

def calculate_trade_statistics(results):
    """Calculate trading statistics from results"""
    if not results:
        return None
        
    # Calculate trade statistics
    total_trades = len(results)
    profitable_trades = sum(1 for trade in results if trade['net_pnl'] > 0)
    unprofitable_trades = sum(1 for trade in results if trade['net_pnl'] <= 0)
    success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_buy_value = sum(trade['matched_buy_value'] for trade in results)
    total_sell_value = sum(trade['matched_sell_value'] for trade in results)
    total_fees = sum(trade['fees'] for trade in results)
    total_pnl = sum(trade['net_pnl'] for trade in results)
    
    return {
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'unprofitable_trades': unprofitable_trades,
        'success_rate': success_rate,
        'total_buy_value': total_buy_value,
        'total_sell_value': total_sell_value,
        'total_fees': total_fees,
        'total_pnl': total_pnl
    }

def export_trade_results(results, start_time, end_time, output_file='trade_results.csv'):
    """Export trade results and statistics to CSV"""
    stats = calculate_trade_statistics(results)
    
    if not stats:
        print("No trade data to export")
        return
    
    # Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header section with overall statistics
        writer.writerow(['Trade Analysis Summary'])
        writer.writerow(['Period Start', start_time.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Period End', end_time.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Trades', stats['total_trades']])
        writer.writerow(['Profitable Trades', stats['profitable_trades']])
        writer.writerow(['Unprofitable Trades', stats['unprofitable_trades']])
        writer.writerow(['Success Rate', f"{stats['success_rate']:.2f}%"])
        writer.writerow(['Total Buy Value', f"{stats['total_buy_value']:.8f}"])
        writer.writerow(['Total Sell Value', f"{stats['total_sell_value']:.8f}"])
        writer.writerow(['Total Fees', f"{stats['total_fees']:.8f}"])
        writer.writerow(['Total Net P&L', f"{stats['total_pnl']:.8f}"])
        writer.writerow([])  # Empty row for separation
        
        # Write detailed trade data
        writer.writerow([
            'Symbol',
            'Buy Value',
            'Sell Value',
            'Realized P&L',
            'Fees',
            'Net P&L'
        ])
        
        # Sort trades by net P&L
        sorted_results = sorted(results, key=lambda x: abs(x['net_pnl']), reverse=True)
        
        for trade in sorted_results:
            writer.writerow([
                trade['symbol'],
                f"{trade['matched_buy_value']:.8f}",
                f"{trade['matched_sell_value']:.8f}",
                f"{trade['realized_pnl']:.8f}",
                f"{trade['fees']:.8f}",
                f"{trade['net_pnl']:.8f}"
            ])
            
    print(f"\nTrade results exported to {output_file}")
    
    # Print summary to console
    print("\n=== Trading Period Summary ===")
    print(f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Profitable Trades: {stats['profitable_trades']}")
    print(f"Unprofitable Trades: {stats['unprofitable_trades']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    print(f"Total Net P&L: {stats['total_pnl']:.8f} USDT")

def load_symbol_data(checkpoint_file):
    """Load trading symbols from checkpoint JSON file"""
    with open(checkpoint_file, 'r') as f:
        symbol_data = json.load(f)
    return list(symbol_data.keys())

def get_trades_for_period(client, symbol, start_time, end_time):
    """Get trades for a specific symbol in given time period by chunking into 24h windows"""
    all_trades = []
    current_start = start_time
    chunk_count = 0
    
    print(f"\nFetching trades for {symbol}")
    print(f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        while current_start < end_time:
            chunk_count += 1
            # Calculate end of current chunk (either 24 hours from start or final end_time)
            chunk_end = min(current_start + timedelta(hours=24), end_time)
            
            # Convert times to milliseconds
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(chunk_end.timestamp() * 1000)
            
            print(f"  Chunk {chunk_count}: {current_start.strftime('%Y-%m-%d %H:%M:%S')} -> {chunk_end.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Fetch trades for current chunk
            chunk_trades = client.get_my_trades(
                symbol=symbol,
                startTime=start_ms,
                endTime=end_ms,
                limit=1000
            )
            
            # Log chunk results
            if chunk_trades:
                print(f"    Found {len(chunk_trades)} trades in chunk {chunk_count}")
                all_trades.extend(chunk_trades)
            else:
                print(f"    No trades found in chunk {chunk_count}")
            
            # Move to next chunk
            current_start = chunk_end
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        print(f"  Total trades found for {symbol}: {len(all_trades)}")
        return all_trades
        
    except Exception as e:
        if 'Invalid symbol' not in str(e):
            print(f"  ERROR fetching trades for {symbol}: {str(e)}")
        return []

def process_symbol_trades(trades):
    """Process trades for a symbol and calculate detailed P&L using FIFO matching"""
    if not trades:
        return None
        
    symbol = trades[0]['symbol']
    buy_queue = []  # Store (qty, price) tuples for buys
    
    # Initialize tracking variables
    matched_buy_value = 0
    matched_sell_value = 0
    matched_buy_qty = 0
    matched_sell_qty = 0
    fees = 0
    
    # Sort trades by time
    sorted_trades = sorted(trades, key=lambda x: x['time'])
    
    print(f"\nDetailed matching for {symbol}:")
    
    # Process each trade
    for trade in sorted_trades:
        qty = float(trade['qty'])
        price = float(trade['price'])
        commission = float(trade['commission'])
        commission_asset = trade['commissionAsset']
        
        # Convert commission to USDT
        if commission_asset != 'USDT' and commission_asset != 'BNB':
            commission_value = commission * price
        else:
            commission_value = commission
        
        fees += commission_value
        
        if trade['isBuyer']:
            # Add buy to queue
            buy_queue.append((qty, price))
            print(f"  Buy: {qty:.8f} @ {price:.8f} USDT")
        else:
            # Process sell against existing buys
            remaining_sell = qty
            sell_value = 0
            matched_buys_value = 0
            matched_qty = 0
            
            while remaining_sell > 0 and buy_queue:
                buy_qty, buy_price = buy_queue[0]
                match_qty = min(buy_qty, remaining_sell)
                
                # Calculate matched values
                buy_value = match_qty * buy_price
                current_sell_value = match_qty * price
                
                matched_buys_value += buy_value
                sell_value += current_sell_value
                matched_qty += match_qty
                
                print(f"  Sell: {match_qty:.8f} @ {price:.8f} USDT (matched with buy @ {buy_price:.8f})")
                
                if match_qty == buy_qty:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_qty - match_qty, buy_price)
                
                remaining_sell -= match_qty
            
            if matched_qty > 0:
                matched_buy_value += matched_buys_value
                matched_sell_value += sell_value
                matched_buy_qty += matched_qty
                matched_sell_qty += matched_qty
    
    # Calculate realized P&L only for matched trades
    realized_pnl = matched_sell_value - matched_buy_value
    
    # Get first and last trade timestamps
    first_trade_time = datetime.fromtimestamp(sorted_trades[0]['time'] / 1000)
    last_trade_time = datetime.fromtimestamp(sorted_trades[-1]['time'] / 1000)
    
    return {
        'symbol': symbol,
        'buy_quantity': matched_buy_qty,
        'sell_quantity': matched_sell_qty,
        'buy_value': matched_buy_value,
        'sell_value': matched_sell_value,
        'realized_pnl': realized_pnl,
        'fees': fees,
        'net_pnl': realized_pnl - fees,
        'matched_buy_qty': matched_buy_qty,
        'matched_buy_value': matched_buy_value,
        'matched_sell_qty': matched_sell_qty,
        'matched_sell_value': matched_sell_value,
        'first_trade_time': first_trade_time,
        'last_trade_time': last_trade_time
    }
    
    return {
        'symbol': symbol,
        'buy_quantity': buy_qty,
        'buy_value': buy_value,
        'sell_quantity': sell_qty,
        'sell_value': sell_value,
        'realized_pnl': realized_pnl,
        'fees': fees,
        'net_pnl': realized_pnl - fees
    }

def main():
    try:
        print("\nCrypto Trading Analysis")
        print("=" * 80)
        
        # Initialize Binance client
        print("Connecting to Binance...")
        client = load_binance_client()
        print("✓ Connected to Binance successfully")
        
        # Load symbols from checkpoint
        print("\nLoading symbol data from checkpoint...")
        symbols = load_symbol_data('crypto_data_checkpoint_20241231.JSON')
        print(f"✓ Loaded {len(symbols)} symbols from checkpoint")
        
        print("\nInitializing analysis parameters...")
        
        # Set time period for last day
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        
        results = []
        total_stats = {
            'buy_value': 0,
            'sell_value': 0,
            'realized_pnl': 0,
            'fees': 0,
            'net_pnl': 0
        }
        
        print("\n=== Last 24 Hours Trading Activity ===")
        
        # Process each symbol
        total_symbols = len(symbols)
        print(f"\nProcessing {total_symbols} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\nProgress: {i}/{total_symbols} ({(i/total_symbols*100):.1f}%)")
            trades = get_trades_for_period(client, symbol, yesterday_start, now)
            
            if trades:
                print(f"Processing trades for {symbol}...")
                result = process_symbol_trades(trades)
                if result and (result['buy_value'] > 0 or result['sell_value'] > 0):
                    results.append(result)
                    # Update totals
                    for key in total_stats:
                        total_stats[key] += result[key]
                    print(f"✓ Found trading activity for {symbol}")
                else:
                    print(f"No significant trading activity for {symbol}")
            else:
                print(f"No trades found for {symbol}")
        
        if results:
            # Print detailed trade information
            print(f"\n{'Symbol':<10} {'Matched Buys':>15} {'Matched Sales':>15} {'Realized P&L':>15} {'Fees':>10} {'Net P&L':>15}")
            print("-" * 80)
            
            # Sort by net P&L
            for result in sorted(results, key=lambda x: abs(x['net_pnl']), reverse=True):
                if result['matched_buy_qty'] > 0:  # Only show symbols with matched trades
                    print(f"{result['symbol']:<10} "
                          f"{result['matched_buy_value']:>15.8f} "
                          f"{result['matched_sell_value']:>15.8f} "
                          f"{result['realized_pnl']:>15.8f} "
                          f"{result['fees']:>10.8f} "
                          f"{result['net_pnl']:>15.8f}")
            
            # Print totals
            print("\n=== Cumulative Totals ===")
            print(f"Total Buy Value: {total_stats['buy_value']:.8f} USDT")
            print(f"Total Sell Value: {total_stats['sell_value']:.8f} USDT")
            print(f"Total Realized P&L: {total_stats['realized_pnl']:.8f} USDT")
            print(f"Total Fees: {total_stats['fees']:.8f} USDT")
            print(f"Total Net P&L: {total_stats['net_pnl']:.8f} USDT")
            
            # Export results to CSV with period information
            export_trade_results(results, yesterday_start, now)
        else:
            print("No trades found in the analysis period")
               
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()