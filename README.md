# Binance Data Collector

A robust, asynchronous Python script for collecting historical cryptocurrency trading data from Binance. This script fetches detailed market data including price history, order book information, and trading metrics for all USDT trading pairs.

## Features

- Asynchronous data collection using `aiohttp`
- Intelligent rate limiting to prevent API bans
- Comprehensive error handling and retry logic
- Memory-efficient chunked data processing
- Detailed logging system with rotating file handlers
- Checkpoint system for resume capability
- Progress tracking with real-time statistics
- Automatic cleanup of old files

## Prerequisites

- Python 3.7+
- Binance API credentials

## Required Python Packages

```bash
pip install python-binance
pandas
python-dotenv
aiohttp
tqdm
psutil
```

## Setup

1. Create a `.env` file in the project root with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

2. Create the following directory structure:
```
project_root/
├── data/
│   ├── symbols/
│   └── chunks/
└── logs/
```

## Usage

Run the script using:
```bash
python script_name.py
```

## Data Collection Process

1. **Initialization**
   - Sets up logging system
   - Creates necessary directories
   - Loads API credentials
   - Fetches list of USDT trading pairs

2. **For Each Trading Pair**
   - Collects daily kline data
   - Fetches 15-minute interval data
   - Retrieves order book information
   - Calculates additional metrics
   - Stores data in chunks for memory efficiency

3. **Data Processing**
   - Processes data in monthly chunks
   - Combines daily and 15-minute data
   - Calculates various trading metrics
   - Saves data in JSON format

## Output Data Structure

The script generates JSON files with the following structure for each symbol:

```json
{
    "date": "YYYY-MM-DD",
    "pair": "SYMBOL",
    "high": float,
    "low": float,
    "open": float,
    "close": float,
    "volume_usdt": float,
    "trades_count": integer,
    "prices_15min": [
        {
            "timestamp": "YYYY-MM-DD HH:MM",
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "quote_volume": float
        }
    ],
    "liquidity": {
        "bid_liquidity": float,
        "ask_liquidity": float,
        "total_liquidity": float,
        "spread": float,
        "spread_percentage": float,
        "best_bid": float,
        "best_ask": float,
        "bid_depth": integer,
        "ask_depth": integer
    },
    "additional_metrics": {
        "volatility": float,
        "taker_buy_ratio": float,
        "average_trade_size": float,
        "price_change": float,
        "volume_weighted_average_price": float,
        "high_low_range": float
    }
}
```

## Error Handling

- Implements exponential backoff for failed requests
- Handles rate limiting with smart delays
- Logs all errors with detailed information
- Maintains checkpoints for recovery

## Memory Management

- Processes data in chunks to minimize memory usage
- Implements garbage collection after processing chunks
- Monitors and logs memory usage
- Cleans up temporary files automatically

## Logging

The script maintains three types of logs:
- Detailed debug logs: `logs/crypto_data_YYYYMMDD_HHMMSS.log`
- Error logs: `logs/errors_YYYYMMDD_HHMMSS.log`
- Console output for important information

## Performance Considerations

- Uses asynchronous requests for improved throughput
- Implements rate limiting to prevent API bans
- Processes data in chunks to manage memory usage
- Includes automatic cleanup of old files

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open-source and available under the MIT License.

## Disclaimer

This script is for educational purposes only. Please ensure you comply with Binance's terms of service and API usage guidelines when using this script.
