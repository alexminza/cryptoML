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

# Initialize session state
if 'monitor_thread' not in st.session_state:
    st.session_state.monitor_thread = None
if 'stop_monitoring' not in st.session_state:
    st.session_state.stop_monitoring = False
if 'orders_config' not in st.session_state:
    st.session_state.orders_config = {}
if 'valid_symbols' not in st.session_state:
    st.session_state.valid_symbols = set()

def initialize_binance():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if api_key and api_secret:
        client = Client(api_key, api_secret)
        # Cache valid trading pairs
        try:
            exchange_info = client.get_exchange_info()
            st.session_state.valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
        except Exception as e:
            st.error(f"Error fetching exchange info: {str(e)}")
        return client
    else:
        st.error("API credentials not found in .env file")
    return None

def is_valid_symbol(symbol):
    return symbol in st.session_state.valid_symbols

def get_positions(client):
    try:
        account = client.get_account()
        positions = []
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                asset = balance['asset']
                if asset != 'USDT':
                    symbol = f"{asset}USDT"
                    # Only process valid trading pairs
                    if is_valid_symbol(symbol):
                        try:
                            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                            positions.append({
                                'asset': asset,
                                'symbol': symbol,
                                'free': free,
                                'locked': locked,
                                'total': free + locked,
                                'current_price': current_price,
                                'value_usdt': (free + locked) * current_price
                            })
                        except Exception as e:
                            st.warning(f"Could not get price for {symbol}: {str(e)}")
        return positions
    except Exception as e:
        st.error(f"Error fetching positions: {str(e)}")
        return []

def get_open_orders(client):
    try:
        all_orders = []
        open_orders = client.get_open_orders()
        
        for order in open_orders:
            symbol = order['symbol']
            if is_valid_symbol(symbol):
                try:
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    
                    order_data = {
                        'symbol': symbol,
                        'orderId': order['orderId'],
                        'side': order['side'],
                        'type': order['type'],
                        'price': float(order['price']),
                        'quantity': float(order['origQty']),
                        'executed': float(order['executedQty']),
                        'remaining': float(order['origQty']) - float(order['executedQty']),
                        'current_price': current_price,
                        'price_diff_percent': ((current_price - float(order['price'])) / float(order['price'])) * 100
                    }
                    all_orders.append(order_data)
                except Exception as e:
                    st.warning(f"Could not process order for {symbol}: {str(e)}")
        return all_orders
    except Exception as e:
        st.error(f"Error fetching open orders: {str(e)}")
        return []

def get_symbol_info(client, symbol):
    try:
        symbol_info = client.get_symbol_info(symbol)
        if symbol_info:
            return {
                'baseAssetPrecision': symbol_info['baseAssetPrecision'],
                'quotePrecision': symbol_info['quotePrecision'],
                'filters': {f['filterType']: f for f in symbol_info['filters']}
            }
        return None
    except Exception as e:
        st.error(f"Error getting symbol info for {symbol}: {str(e)}")
        return None

def monitor_positions(client):
    while not st.session_state.stop_monitoring:
        try:
            positions = get_positions(client)
            open_orders = get_open_orders(client)
            
            for position in positions:
                symbol = position['symbol']
                if not is_valid_symbol(symbol):
                    continue
                    
                config = st.session_state.orders_config.get(symbol, {})
                
                if config.get('stop_loss_price'):
                    current_price = position['current_price']
                    stop_loss_price = config['stop_loss_price']
                    
                    if current_price <= stop_loss_price:
                        try:
                            # First cancel existing sell limit order
                            if config.get('sell_order_id'):
                                client.cancel_order(
                                    symbol=symbol,
                                    orderId=config['sell_order_id']
                                )
                            
                            # Then execute market sell for stop loss
                            quantity = config.get('position_quantity', position['free'])
                            symbol_info = get_symbol_info(client, symbol)
                            if symbol_info:
                                quantity = round(float(quantity), symbol_info['baseAssetPrecision'])
                                
                                if quantity > 0:
                                    client.create_order(
                                        symbol=symbol,
                                        side=SIDE_SELL,
                                        type=ORDER_TYPE_MARKET,
                                        quantity=quantity
                                    )
                                    # Clear the configuration after stop loss
                                    st.session_state.orders_config.pop(symbol, None)
                                    st.error(f"Stop loss triggered for {symbol} at {current_price}")
                        except Exception as e:
                            st.error(f"Error executing stop loss for {symbol}: {str(e)}")
            
            # Check for completed buy orders and place sell orders
            for order in open_orders:
                if order['side'] == 'BUY' and order['executed'] > 0:
                    config = st.session_state.orders_config.get(order['symbol'], {})
                    if config.get('sell_price') and not config.get('sell_order_placed'):
                        try:
                            symbol_info = get_symbol_info(client, order['symbol'])
                            if symbol_info:
                                quantity = round(order['executed'], symbol_info['baseAssetPrecision'])
                                
                                if quantity > 0:
                                    client.create_order(
                                        symbol=order['symbol'],
                                        side=SIDE_SELL,
                                        type=ORDER_TYPE_LIMIT,
                                        timeInForce=TIME_IN_FORCE_GTC,
                                        quantity=quantity,
                                        price=str(config['sell_price'])
                                    )
                                    st.session_state.orders_config[order['symbol']]['sell_order_placed'] = True
                        except Exception as e:
                            st.error(f"Error placing sell order for {order['symbol']}: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in monitor thread: {str(e)}")
        
        time.sleep(10)  # Check every 10 seconds

def start_monitoring():
    if st.session_state.monitor_thread is None or not st.session_state.monitor_thread.is_alive():
        st.session_state.stop_monitoring = False
        client = initialize_binance()
        if client:
            st.session_state.monitor_thread = threading.Thread(target=monitor_positions, args=(client,))
            st.session_state.monitor_thread.start()
            st.success("Position monitoring started")

def stop_monitoring():
    st.session_state.stop_monitoring = True
    if st.session_state.monitor_thread and st.session_state.monitor_thread.is_alive():
        st.session_state.monitor_thread.join()
    st.session_state.monitor_thread = None
    st.warning("Position monitoring stopped")

def main():
    st.title("Binance Trade Manager")
    
    client = initialize_binance()
    
    if client:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Monitoring"):
                start_monitoring()
        with col2:
            if st.button("Stop Monitoring"):
                stop_monitoring()
        with col3:
            if st.button("Refresh Data"):
                st.rerun()
                
        tab1, tab2 = st.tabs(["Active Positions", "Open Orders"])
        
        with tab1:
            st.subheader("Active Positions")
            positions = get_positions(client)
            
            if positions:
                for position in positions:
                    with st.expander(f"{position['symbol']} - {position['total']} {position['asset']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"Free: {position['free']:.8f}")
                            st.write(f"Locked: {position['locked']:.8f}")
                            st.write(f"Current Price: ${position['current_price']:.4f}")
                            st.write(f"Total Value: ${position['value_usdt']:.2f}")
                            
                            # Order configuration inputs
                            sell_price = st.number_input(
                                "Sell Limit Price",
                                min_value=0.0,
                                value=float(position['current_price'] * 1.02),
                                key=f"sell_price_{position['symbol']}"
                            )
                            
                            stop_loss_percent = st.number_input(
                                "Stop Loss (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=2.0,
                                key=f"stop_loss_{position['symbol']}"
                            )
                            
                            if st.button("Set Orders", key=f"set_orders_{position['symbol']}"):
                                try:
                                    # Get symbol info for precision
                                    symbol_info = get_symbol_info(client, position['symbol'])
                                    if symbol_info:
                                        quantity = round(position['free'], symbol_info['baseAssetPrecision'])
                                        
                                        # Place sell limit order immediately
                                        sell_order = client.create_order(
                                            symbol=position['symbol'],
                                            side=SIDE_SELL,
                                            type=ORDER_TYPE_LIMIT,
                                            timeInForce=TIME_IN_FORCE_GTC,
                                            quantity=quantity,
                                            price=str(sell_price)
                                        )
                                        
                                        stop_loss_price = position['current_price'] * (1 - stop_loss_percent / 100)
                                        st.session_state.orders_config[position['symbol']] = {
                                            'sell_price': sell_price,
                                            'stop_loss_price': stop_loss_price,
                                            'sell_order_id': sell_order['orderId'],
                                            'position_quantity': quantity
                                        }
                                        st.success(f"Sell limit order placed for {position['symbol']}")
                                except Exception as e:
                                    st.error(f"Error placing sell order: {str(e)}")
                        
                        with col2:
                            # Show current configuration if exists
                            config = st.session_state.orders_config.get(position['symbol'], {})
                            if config:
                                st.subheader("Current Configuration")
                                st.write(f"Sell Price: ${config.get('sell_price', 'Not set')}")
                                st.write(f"Stop Loss Price: ${config.get('stop_loss_price', 'Not set')}")
                                st.write(f"Sell Order Placed: {'Yes' if config.get('sell_order_placed') else 'No'}")
            else:
                st.info("No active positions found")
        
        with tab2:
            st.subheader("Open Orders")
            open_orders = get_open_orders(client)
            
            if open_orders:
                for order in open_orders:
                    with st.expander(f"{order['symbol']} - {order['side']} Order"):
                        st.write(f"Order ID: {order['orderId']}")
                        st.write(f"Type: {order['type']}")
                        st.write(f"Price: ${order['price']:.4f}")
                        st.write(f"Current Price: ${order['current_price']:.4f}")
                        st.write(f"Quantity: {order['quantity']:.8f}")
                        st.write(f"Executed: {order['executed']:.8f}")
                        st.write(f"Remaining: {order['remaining']:.8f}")
                        st.write(f"Price Difference: {order['price_diff_percent']:.2f}%")
            else:
                st.info("No open orders found")

if __name__ == "__main__":
    main()