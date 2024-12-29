# Neural Network-Powered Trading Agent

A sophisticated trading agent that uses deep learning to generate short-term trading signals for cryptocurrency markets. The project consists of three main components: data collection, model training, and trading execution.

## Project Structure

```
Currently under refactoring
use model_componets.py
```

## Features and Strategy

### Feature Engineering

The model uses a comprehensive set of features designed to capture different aspects of market behavior:

1. **Market Structure Features**
   - Depth imbalance (bid/ask ratio)
   - Total liquidity (logarithmic scale)
   - Order book depth analysis

2. **Market Dynamics**
   - 15-minute volatility
   - Volume momentum (normalized against 24h average)
   - Price returns (clipped to handle outliers)

3. **Price Action Indicators**
   - VWAP deviation
   - 15-minute price momentum
   - Bollinger Band position

4. **Technical Indicators**
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands analysis

### Trading Strategy Triggers

The strategy employs a multi-factor approach for trade signals:

1. **Entry Conditions**
   - Model confidence > 0.6
   - Volume momentum > 0.8
   - RSI < 75 (not overbought)
   - Price position within Bollinger Bands < 0.85

2. **Risk Management**
   - Dynamic stop-loss based on volatility
   - Take-profit targets at 1.5% gain
   - Position sizing based on account risk parameters

## Components

### 1. Data Collection (binance_data_detailed.py)

A robust data collection system that:
- Fetches historical and real-time market data from Binance
- Implements intelligent rate limiting and error handling
- Processes and stores data efficiently with automated cleanup
- Features include:
  - Async processing for improved performance
  - Atomic file operations for data integrity
  - Automated memory management
  - Detailed logging system

### 2. Model Training (models_components.py)

Advanced neural network architecture with:
- Adaptive model complexity based on asset type
- Robust feature scaling and preprocessing
- Enhanced training process with:
  - Dynamic learning rate scheduling
  - Early stopping with patience
  - Class imbalance handling
  - Cross-validation for stability

### 3. Trading UI (binance_trading_bot.py)

Streamlit-based interface providing:
- Real-time position monitoring
- Order management system
- Risk management tools
- Performance analytics

## Setup and Installation

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. **Configuration**
Create a `.env` file with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Future Development

- Migration to FastAPI for improved scalability
- Enhanced backtesting capabilities
- Additional feature engineering
- Portfolio management optimization
- Real-time market analysis dashboard

## Note

This project is currently a work in progress. While the core functionality is implemented, some features are under development and optimization is ongoing.

## Requirements
- Python 3.8+
- Binance account with API access
- Required packages listed in requirements.txt

## Usage

1. **Data Collection**
```bash
python binance_data_detailed.py
```

2. **Model Training**
```bash
python models_components.py
```

3. **Trading Interface**
```bash
streamlit run binance_trading_bot.py
```

## Warning

Trading cryptocurrencies involves significant risk of loss and is not suitable for all investors. This software is for educational purposes only and does not constitute financial advice.