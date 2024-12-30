from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import json
from dotenv import load_dotenv
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import aiohttp
from typing import List, Dict, Optional
import random
from collections import deque
import hmac
import hashlib
from tqdm import tqdm
import gc
import psutil
import sys

def setup_logging():
    """Configure detailed logging with both file and console handlers"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger('crypto_data')
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler for everything
    file_handler = RotatingFileHandler(
        f'logs/crypto_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Error log file for warnings and errors
    error_handler = RotatingFileHandler(
        f'logs/errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger

logger = setup_logging()

def get_symbol_filepath(symbol: str) -> str:
    """Generate filepath for individual symbol data"""
    return f'data/symbols/{symbol}_{datetime.now().strftime("%Y%m%d")}.json'

def cleanup_old_files(directory: str, days: int = 7):
    """Remove files older than specified days"""
    current_time = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.stat(filepath).st_mtime < current_time - (days * 86400):
                os.remove(filepath)

def cleanup_temp_files():
    """Remove all temporary .temp files"""
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.temp'):
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    logger.error(f"Error removing temp file {file}: {str(e)}")

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def safe_json_dump(data, filepath: str) -> bool:
    """Safely dump data to JSON file with atomic write"""
    try:
        temp_file = f"{filepath}.temp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, filepath)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

class DataCollectionStats:
    def __init__(self):
        self.start_time = time.time()
        self.requests_made = 0
        self.rate_limits_hit = 0
        self.successful_pairs = 0
        self.failed_pairs = 0
        self.current_pair = ""
        self.total_pairs = 0
        self.last_stats_print = 0
        self.print_interval = 60  # Print stats every 60 seconds
        self.peak_memory = 0

    def update_memory_stats(self):
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def log_stats(self, force=False):
        current_time = time.time()
        if force or (current_time - self.last_stats_print) >= self.print_interval:
            elapsed_time = current_time - self.start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            self.update_memory_stats()
            
            logger.info(
                f"Stats: Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}, "
                f"Requests: {self.requests_made}, "
                f"Rate Limits: {self.rate_limits_hit}, "
                f"Success: {self.successful_pairs}/{self.total_pairs}, "
                f"Current: {self.current_pair}, "
                f"Memory: {self.peak_memory:.1f}MB"
            )
            self.last_stats_print = current_time

stats = DataCollectionStats()

class AsyncRateLimiter:
    def __init__(self):
        self.minute_request_count = 0
        self.last_minute_reset = time.time()
        self.request_weights = deque()
        self._lock = asyncio.Lock()
        self.MAX_REQUESTS_PER_MINUTE = 400  # More conservative limit
        self.MIN_REQUEST_INTERVAL = 0.25  # 250ms between requests
        logger.info(f"Rate limiter initialized with {self.MAX_REQUESTS_PER_MINUTE} requests/minute limit")

    async def acquire(self, weight: int = 1):
        async with self._lock:
            current_time = time.time()
            
            # Reset minute counter if a minute has passed
            if current_time - self.last_minute_reset >= 60:
                logger.debug(f"Resetting minute counter. Previous count: {self.minute_request_count}")
                self.minute_request_count = 0
                self.last_minute_reset = current_time
                self.request_weights.clear()

            # Remove old request weights
            while self.request_weights and current_time - self.request_weights[0][1] >= 60:
                old_weight, _ = self.request_weights.popleft()
                self.minute_request_count -= old_weight

            # If we're at 70% of the limit, add extra delay
            if self.minute_request_count > (self.MAX_REQUESTS_PER_MINUTE * 0.70):
                delay = random.uniform(0.5, 1.5)
                logger.debug(f"Approaching rate limit ({self.minute_request_count} requests), adding {delay:.2f}s delay")
                await asyncio.sleep(delay)

            # If we're close to the limit, wait for the next minute
            if self.minute_request_count + weight > self.MAX_REQUESTS_PER_MINUTE:
                wait_time = 61 - (current_time - self.last_minute_reset)  # Add 1 second buffer
                logger.warning(f"Rate limit reached ({self.minute_request_count}), waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.minute_request_count = 0
                self.last_minute_reset = time.time()
                self.request_weights.clear()

            # Add minimum delay between requests
            if self.request_weights:
                last_request_time = self.request_weights[-1][1]
                time_since_last_request = current_time - last_request_time
                if time_since_last_request < self.MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(self.MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, 0.1))

            self.minute_request_count += weight
            self.request_weights.append((weight, time.time()))
            stats.requests_made += 1

class ChunkManager:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.current_chunk = []
        self.chunk_count = 0

    def add_item(self, item):
        self.current_chunk.append(item)
        if len(self.current_chunk) >= self.chunk_size:
            self.save_chunk()

    def save_chunk(self):
        if not self.current_chunk:
            return
        
        filename = f'data/chunks/chunk_{self.chunk_count}.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if safe_json_dump(self.current_chunk, filename):
            logger.debug(f"Saved chunk {self.chunk_count} with {len(self.current_chunk)} items")
            self.chunk_count += 1
            self.current_chunk = []
            gc.collect()

    def finalize(self):
        self.save_chunk()  # Save any remaining items
        return self.chunk_count

class BinanceDataFetcher:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limiter = AsyncRateLimiter()
        self.session = None
        self.headers = {'X-MBX-APIKEY': self.api_key}
        self.retry_delay = 1
        self.max_retries = 3
        logger.info("BinanceDataFetcher initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_with_retry(self, url: str, params: Dict, weight: int = 1) -> Optional[Dict]:
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire(weight)
                async with self.session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limit hit for {params.get('symbol', 'unknown')}, waiting {retry_after}s")
                        stats.rate_limits_hit += 1
                        await asyncio.sleep(retry_after + 1)  # Add 1 second buffer
                        continue
                    
                    if response.status == 418:  # IP ban
                        logger.error("IP has been auto-banned for repeated rate limit violations")
                        raise Exception("IP banned by Binance")
                    
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed for {params.get('symbol', 'unknown')}, attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Final request attempt failed for {params.get('symbol', 'unknown')}: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
        
        return None

    async def fetch_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List:
        logger.debug(f"Fetching {interval} klines for {symbol}")
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        data = await self.fetch_with_retry("https://api.binance.com/api/v3/klines", params, weight=2)
        
        # Clear memory
        params = None
        gc.collect()
        
        return data

    async def fetch_order_book(self, symbol: str) -> Dict:
        logger.debug(f"Fetching order book for {symbol}")
        params = {'symbol': symbol, 'limit': 1000}
        data = await self.fetch_with_retry("https://api.binance.com/api/v3/depth", params, weight=10)
        
        if data:
            try:
                result = {}
                
                # Process bids and asks in chunks
                bid_chunks = [data['bids'][i:i+10] for i in range(0, 100, 10)]
                ask_chunks = [data['asks'][i:i+10] for i in range(0, 100, 10)]
                
                bid_liquidity = 0
                ask_liquidity = 0
                
                for chunk in bid_chunks:
                    bid_liquidity += sum(float(bid[1]) for bid in chunk)
                for chunk in ask_chunks:
                    ask_liquidity += sum(float(ask[1]) for ask in chunk)
                
                spread = float(data['asks'][0][0]) - float(data['bids'][0][0])
                
                result = {
                    'bid_liquidity': bid_liquidity,
                    'ask_liquidity': ask_liquidity,
                    'total_liquidity': bid_liquidity + ask_liquidity,
                    'spread': spread,
                    'spread_percentage': (spread / float(data['bids'][0][0])) * 100,
                    'best_bid': float(data['bids'][0][0]),
                    'best_ask': float(data['asks'][0][0]),
                    'bid_depth': len(data['bids']),
                    'ask_depth': len(data['asks'])
                }
                
                # Clear memory
                data = None
                bid_chunks = None
                ask_chunks = None
                gc.collect()
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing order book data for {symbol}: {str(e)}")
                return None
        return None

async def process_daily_data(fetcher: BinanceDataFetcher, symbol: str, day_kline: List, chunk_manager: ChunkManager):
    try:
        day_date = datetime.fromtimestamp(day_kline[0] / 1000)
        
        # Fetch 15-minute data for this day
        day_start_ts = int(day_date.timestamp() * 1000)
        day_end_ts = int((day_date + timedelta(days=1)).timestamp() * 1000)
        
        fifteen_min_klines = await fetcher.fetch_klines(symbol, '15m', day_start_ts, day_end_ts)
        await asyncio.sleep(0.5)
        
        liquidity_data = await fetcher.fetch_order_book(symbol)

        if not fifteen_min_klines:
            return False

        # Process 15-minute data in chunks to save memory
        fifteen_min_prices = []
        for i in range(0, len(fifteen_min_klines), 10):
            chunk = fifteen_min_klines[i:i+10]
            chunk_data = [
                {
                    'timestamp': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d %H:%M'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'quote_volume': float(k[7])
                }
                for k in chunk
            ]
            fifteen_min_prices.extend(chunk_data)
            del chunk_data
            gc.collect()

        day_data = {
            'date': day_date.strftime('%Y-%m-%d'),
            'pair': symbol,
            'high': float(day_kline[2]),
            'low': float(day_kline[3]),
            'open': float(day_kline[1]),
            'close': float(day_kline[4]),
            'volume_usdt': float(day_kline[7]),
            'trades_count': int(day_kline[8]),
            'prices_15min': fifteen_min_prices,
            'liquidity': liquidity_data,
            'additional_metrics': {
                'volatility': (float(day_kline[2]) - float(day_kline[3])) / float(day_kline[3]) * 100,
                'taker_buy_ratio': float(day_kline[10]) / float(day_kline[7]) if float(day_kline[7]) > 0 else 0,
                'average_trade_size': float(day_kline[7]) / int(day_kline[8]) if int(day_kline[8]) > 0 else 0,
                'price_change': ((float(day_kline[4]) - float(day_kline[1])) / float(day_kline[1])) * 100,
                'volume_weighted_average_price': float(day_kline[7]) / float(day_kline[5]) if float(day_kline[5]) > 0 else 0,
                'high_low_range': float(day_kline[2]) - float(day_kline[3])
            }
        }

        # Add to chunk manager
        chunk_manager.add_item(day_data)

        # Clear memory
        del fifteen_min_prices
        del day_data
        gc.collect()

        return True

    except Exception as e:
        logger.error(f"Error processing day data for {symbol}: {str(e)}")
        return False

async def process_month_data(fetcher: BinanceDataFetcher, symbol: str, start_date: datetime, end_date: datetime, chunk_manager: ChunkManager):
    logger.debug(f"Processing month data for {symbol}: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    try:
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        daily_klines = await fetcher.fetch_klines(symbol, '1d', start_ts, end_ts)
        if not daily_klines:
            return False

        success = True
        for day_kline in daily_klines:
            if not await process_daily_data(fetcher, symbol, day_kline, chunk_manager):
                success = False
            await asyncio.sleep(0.2)
            gc.collect()

        return success

    except Exception as e:
        logger.error(f"Error processing month data for {symbol}: {str(e)}")
        return False

async def process_symbol(fetcher: BinanceDataFetcher, symbol: str, start_date: datetime, end_date: datetime):
    logger.info(f"Starting processing of {symbol}")
    stats.current_pair = symbol
    
    try:
        current_date = start_date
        chunk_manager = ChunkManager(chunk_size=50)  # Store 50 days worth of data per chunk
        
        success = True
        while current_date < end_date:
            if current_date.month == 12:
                next_month = datetime(current_date.year + 1, 1, 1)
            else:
                next_month = datetime(current_date.year, current_date.month + 1, 1)
            
            month_end = min(next_month, end_date)
            logger.info(f"Processing {symbol} for {current_date.strftime('%Y-%m')}")
            
            if not await process_month_data(fetcher, symbol, current_date, month_end, chunk_manager):
                success = False
                break
            
            current_date = next_month
            await asyncio.sleep(1)
            gc.collect()
            
            stats.log_stats()
            log_memory_usage()

        # Finalize chunks
        chunk_manager.finalize()

        if success:
            stats.successful_pairs += 1
            logger.info(f"Successfully completed processing {symbol}")
            return symbol, True
        else:
            stats.failed_pairs += 1
            logger.error(f"No data collected for {symbol}")
            return symbol, False

    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        stats.failed_pairs += 1
        return symbol, False

async def combine_chunk_files(symbol: str) -> Optional[List]:
    """Combine all chunk files for a symbol into a single list"""
    try:
        combined_data = []
        chunk_dir = 'data/chunks'
        chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.json')]
        
        for chunk_file in chunk_files:
            with open(os.path.join(chunk_dir, chunk_file), 'r') as f:
                chunk_data = json.load(f)
                combined_data.extend(chunk_data)
            
            # Remove chunk file after processing
            os.remove(os.path.join(chunk_dir, chunk_file))
            gc.collect()
        
        return combined_data
    except Exception as e:
        logger.error(f"Error combining chunks for {symbol}: {str(e)}")
        return None

async def save_symbol_data(symbol: str, data: List):
    """Save symbol data to its final file"""
    try:
        symbol_file = get_symbol_filepath(symbol)
        if safe_json_dump(data, symbol_file):
            logger.info(f"Successfully saved data for {symbol}")
            return symbol_file
        return None
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {str(e)}")
        return None

async def main():
    # Initialize directories and cleanup
    os.makedirs('data/symbols', exist_ok=True)
    os.makedirs('data/chunks', exist_ok=True)
    cleanup_old_files('data/symbols')
    cleanup_old_files('data/chunks')
    cleanup_temp_files()
    
    logger.info("Starting crypto data collection process")
    log_memory_usage()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logger.error("API credentials not found in .env file")
        raise ValueError("API credentials not found in .env file")

    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Get trading pairs
    try:
        client = Client(api_key, api_secret)
        exchange_info = client.get_exchange_info()
        trading_pairs = [s['symbol'] for s in exchange_info['symbols'] 
                        if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
        stats.total_pairs = len(trading_pairs)
        logger.info(f"Found {len(trading_pairs)} USDT trading pairs")
    except Exception as e:
        logger.error(f"Error getting trading pairs: {str(e)}")
        raise

    # Load checkpoint
    checkpoint_file = f'crypto_data_checkpoint_{datetime.now().strftime("%Y%m%d")}.json'
    processed_symbols = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_symbols = set(checkpoint_data.keys())
            logger.info(f"Loaded checkpoint with {len(processed_symbols)} processed pairs")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")

    async with BinanceDataFetcher(api_key, api_secret) as fetcher:
        progress_bar = tqdm(total=len(trading_pairs), desc="Processing pairs")
        
        for symbol in trading_pairs:
            try:
                if symbol in processed_symbols:
                    logger.info(f"Skipping already processed symbol: {symbol}")
                    progress_bar.update(1)
                    continue
                
                # Process symbol and get chunks
                result = await process_symbol(fetcher, symbol, start_date, end_date)
                if result[1]:  # If processing was successful
                    # Combine chunks and save final data
                    combined_data = await combine_chunk_files(symbol)
                    if combined_data:
                        symbol_file = await save_symbol_data(symbol, combined_data)
                        if symbol_file:
                            # Update checkpoint
                            checkpoint_data = {}
                            if os.path.exists(checkpoint_file):
                                with open(checkpoint_file, 'r') as f:
                                    checkpoint_data = json.load(f)
                            
                            checkpoint_data[symbol] = symbol_file
                            safe_json_dump(checkpoint_data, checkpoint_file)
                            
                            logger.info(f"Saved {symbol} data and updated checkpoint")

                progress_bar.update(1)
                stats.log_stats(force=True)
                log_memory_usage()
                
                # Force garbage collection
                gc.collect()
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                continue

        progress_bar.close()

        # Combine all data if needed (optional)
        if input("Combine all data into single file? (y/n): ").lower() == 'y':
            output_filename = f'crypto_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            logger.info("Starting to combine all data files...")
            
            try:
                with open(output_filename, 'w') as out_f:
                    out_f.write('{')
                    
                    symbol_files = [f for f in os.listdir('data/symbols') if f.endswith('.json')]
                    for i, symbol_file in enumerate(symbol_files):
                        symbol = symbol_file.split('_')[0]
                        
                        with open(f'data/symbols/{symbol_file}', 'r') as f:
                            if i > 0:
                                out_f.write(',')
                            out_f.write(f'"{symbol}":')
                            out_f.write(f.read())
                        
                        gc.collect()
                        
                    out_f.write('}')
                logger.info(f"Data successfully combined into {output_filename}")
            except Exception as e:
                logger.error(f"Error combining final data: {str(e)}")

        # Log final statistics
        logger.info(f"""
            Data collection completed:
            Total pairs processed: {stats.successful_pairs}/{stats.total_pairs}
            Failed pairs: {stats.failed_pairs}
            Total requests made: {stats.requests_made}
            Rate limits hit: {stats.rate_limits_hit}
            Peak memory usage: {stats.peak_memory:.1f} MB
            Total runtime: {time.time() - stats.start_time:.1f} seconds
                    """)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        stats.log_stats(force=True)
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        raise