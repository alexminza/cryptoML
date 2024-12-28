import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import ta
import time
from datetime import datetime, timedelta
import threading
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Streamlit configuration
st.set_page_config(page_title="Binance Trade Manager", layout="wide")

def initialize_binance():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if api_key and api_secret:
        return Client(api_key, api_secret)
    else:
        st.error("API credentials not found in .env file")
    return None

def get_technical_indicators(symbol, client):
    try:
        # Get historical klines/candlestick data
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
        
        if not klines:
            return None

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                         'volume', 'close_time', 'quote_asset_volume', 
                                         'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        df['VWAP_Deviation'] = ((df['close'] - df['VWAP']) / df['VWAP']) * 100
        
        df['Volatility'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).average_true_range()
        
        # Volume momentum
        df['Volume_MA'] = df['volume'].rolling(window=24).mean()
        df['Volume_Momentum'] = (df['volume'] - df['Volume_MA']) / df['Volume_MA'] * 100
        
        return df.iloc[-1]
    except Exception as e:
        st.error(f"Error calculating indicators for {symbol}: {str(e)}")
        return None

def get_all_open_positions(client):
    try:
        # Get account information
        account = client.get_account()
        
        # Filter non-zero balances
        positions = []
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                asset = balance['asset']
                if asset != 'USDT':  # Exclude USDT
                    positions.append({
                        'asset': asset,
                        'symbol': f"{asset}USDT",
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    })
        
        return positions
    except Exception as e:
        st.error(f"Error fetching positions: {str(e)}")
        return []

def create_sell_order(client, symbol, quantity, price):
    try:
        order = client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=str(price)
        )
        return True, "Order placed successfully"
    except Exception as e:
        return False, f"Error placing order: {str(e)}"

def main():
    st.title("Binance Portfolio Manager")
    
    client = initialize_binance()
    
    if client:
        # Get all positions
        positions = get_all_open_positions(client)
        
        if positions:
            st.subheader("Your Active Positions")
            
            # Create a table for all positions and their metrics
            position_data = []
            
            for pos in positions:
                symbol = pos['symbol']
                current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                indicators = get_technical_indicators(symbol, client)
                
                if indicators is not None:
                    position_data.append({
                        'Symbol': symbol,
                        'Holdings': f"{pos['total']:.4f}",
                        'Current Price': f"${current_price:.4f}",
                        'Value (USDT)': f"${pos['total'] * current_price:.2f}",
                        'RSI': f"{indicators['RSI']:.1f}",
                        'BB Position': f"{indicators['BB_position']:.2f}",
                        'VWAP Dev %': f"{indicators['VWAP_Deviation']:.2f}%",
                        'Vol Momentum': f"{indicators['Volume_Momentum']:.2f}%",
                        'Volatility': f"{indicators['Volatility']:.4f}"
                    })
            
            # Display positions table
            df = pd.DataFrame(position_data)
            st.dataframe(df, use_container_width=True)
            
            # Create tabs for each position
            tabs = st.tabs([pos['symbol'] for pos in positions])
            
            for i, (tab, pos) in enumerate(zip(tabs, positions)):
                with tab:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Position Details")
                        st.write(f"Free: {pos['free']}")
                        st.write(f"Locked: {pos['locked']}")
                        
                        # Get current price
                        current_price = float(client.get_symbol_ticker(symbol=pos['symbol'])['price'])
                        st.write(f"Current Price: ${current_price:.4f}")
                        
                        # Sell order form
                        st.subheader("Create Sell Order")
                        sell_price = st.number_input(
                            "Sell Price (USDT)",
                            min_value=0.0,
                            value=float(current_price),
                            step=0.0001,
                            key=f"sell_price_{i}"
                        )
                        
                        sell_quantity = st.number_input(
                            "Quantity",
                            min_value=0.0,
                            max_value=float(pos['free']),
                            value=float(pos['free']),
                            step=0.0001,
                            key=f"quantity_{i}"
                        )
                        
                        if st.button("Place Sell Order", key=f"sell_button_{i}"):
                            success, message = create_sell_order(
                                client,
                                pos['symbol'],
                                sell_quantity,
                                sell_price
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                    with col2:
                        st.subheader("Technical Analysis")
                        indicators = get_technical_indicators(pos['symbol'], client)
                        if indicators is not None:
                            metrics = {
                                "RSI": [indicators['RSI'], "Oversold < 30, Overbought > 70"],
                                "BB Position": [indicators['BB_position'], "0 = Lower Band, 1 = Upper Band"],
                                "VWAP Deviation": [indicators['VWAP_Deviation'], "% deviation from VWAP"],
                                "Volume Momentum": [indicators['Volume_Momentum'], "% above/below 24h MA"],
                                "Volatility (ATR)": [indicators['Volatility'], "Higher = More Volatile"]
                            }
                            
                            for metric, (value, description) in metrics.items():
                                st.metric(
                                    label=f"{metric} ({description})",
                                    value=f"{value:.2f}"
                                )
        
        else:
            st.info("No open positions found. Your positions will appear here once you have active trades.")
    
    # Refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()