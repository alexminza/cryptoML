import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import ta
import time
import math
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

def round_down_step_size(quantity: float, step_size: float) -> float:
    """Round down quantity to step size precision"""
    precision = int(round(-math.log(step_size, 10)))
    return math.floor(quantity * (1 / step_size)) / (1 / step_size)

def round_price_precision(price: float, tick_size: float) -> float:
    """Round price to tick size precision"""
    precision = int(round(-math.log(tick_size, 10)))
    return float(round(price * (1 / tick_size)) / (1 / tick_size))

def initialize_binance():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        st.error("API credentials not found in .env file")
        return None
        
    client = Client(api_key, api_secret)
    
    # Test API permissions
    try:
        # Test account access
        client.get_account()
        
        # Cache valid trading pairs
        exchange_info = client.get_exchange_info()
        st.session_state.valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
        
        # Test order creation permission with a fake order
        try:
            client.create_test_order(
                symbol='BTCUSDT',
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=0.001,
                price='1.0'
            )
        except Exception as e:
            if 'APIError(code=-2015)' in str(e):
                st.error("⚠️ API Key does not have trading permissions. Please enable Spot & Margin Trading in your Binance API settings.")
                return None
            
        return client
        
    except Exception as e:
        if 'APIError(code=-2015)' in str(e):
            st.error("⚠️ Invalid API key or insufficient permissions. Please check your API key settings in Binance.")
        else:
            st.error(f"Error initializing Binance client: {str(e)}")
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
            filters = {f['filterType']: f for f in symbol_info['filters']}
            
            # Get price filter
            price_filter = filters.get('PRICE_FILTER', {})
            tick_size = float(price_filter.get('tickSize', 0.00000001))
            
            # Get lot size filter
            lot_filter = filters.get('LOT_SIZE', {})
            step_size = float(lot_filter.get('stepSize', 0.00000001))
            
            return {
                'baseAssetPrecision': symbol_info['baseAssetPrecision'],
                'quotePrecision': symbol_info['quotePrecision'],
                'filters': filters,
                'tick_size': tick_size,
                'step_size': step_size
            }
        return None
    except Exception as e:
        st.error(f"Error getting symbol info for {symbol}: {str(e)}")
        return None

def round_step_size(quantity: float, step_size: float) -> float:
    """Round quantity to step size precision"""
    precision = int(round(-math.log(step_size, 10)))
    return float(round(quantity * (1 / step_size)) / (1 / step_size))

def round_price_precision(price: float, tick_size: float) -> float:
    """Round price to tick size precision"""
    precision = int(round(-math.log(tick_size, 10)))
    return float(round(price * (1 / tick_size)) / (1 / tick_size))

def validate_and_format_order(symbol_info, quantity, price):
    """Validate and format order quantity and price according to symbol rules"""
    if not symbol_info:
        raise ValueError("Symbol info not available")
        
    # Round quantity to valid step size
    formatted_quantity = round_step_size(quantity, symbol_info['step_size'])
    
    # Round price to valid tick size
    formatted_price = round_price_precision(price, symbol_info['tick_size'])
    
    # Get price filter
    price_filter = symbol_info['filters'].get('PRICE_FILTER', {})
    min_price = float(price_filter.get('minPrice', 0))
    max_price = float(price_filter.get('maxPrice', float('inf')))
    
    # Get lot size filter
    lot_filter = symbol_info['filters'].get('LOT_SIZE', {})
    min_qty = float(lot_filter.get('minQty', 0))
    max_qty = float(lot_filter.get('maxQty', float('inf')))
    
    # Validate price
    if formatted_price < min_price or formatted_price > max_price:
        raise ValueError(f"Price {formatted_price} is outside allowed range [{min_price}, {max_price}]")
    
    # Validate quantity
    if formatted_quantity < min_qty or formatted_quantity > max_qty:
        raise ValueError(f"Quantity {formatted_quantity} is outside allowed range [{min_qty}, {max_qty}]")
        
    return formatted_quantity, formatted_price

def monitor_positions(client):
    while not st.session_state.stop_monitoring:
        try:
            positions = get_positions(client)
            
            for position in positions:
                symbol = position['symbol']
                if not is_valid_symbol(symbol):
                    continue
                    
                config = st.session_state.orders_config.get(symbol, {})
                if config and config.get('order_type') == 'TAKE_PROFIT':
                    current_price = position['current_price']
                    stop_loss_price = config.get('stop_loss_price')
                    
                    if current_price <= stop_loss_price:
                        try:
                            # First cancel the take profit order
                            client.cancel_order(
                                symbol=symbol,
                                orderId=config['sell_order_id']
                            )
                            
                            # Get latest symbol info for precision
                            symbol_info = get_symbol_info(client, symbol)
                            if symbol_info:
                                quantity, _ = validate_and_format_order(
                                    symbol_info,
                                    position['free'],
                                    current_price  # Price not used for market order
                                )
                                
                                # Execute market sell for stop loss
                                if quantity > 0:
                                    client.create_order(
                                        symbol=symbol,
                                        side=SIDE_SELL,
                                        type=ORDER_TYPE_MARKET,
                                        quantity=quantity
                                    )
                                    st.session_state.orders_config.pop(symbol, None)
                                    st.error(f"Stop loss triggered for {symbol} at {current_price}")
                                    
                        except Exception as e:
                            st.error(f"Error executing stop loss for {symbol}: {str(e)}")
            
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
                                        # Round down the available quantity according to LOT_SIZE
                                        available_quantity = float(position['free'])
                                        quantity = round_down_step_size(available_quantity, symbol_info['step_size'])
                                        
                                        if quantity <= 0:
                                            st.error(f"Order quantity too small for {position['symbol']}")
                                            return

                                        # Format the price according to rules
                                        formatted_price = round_price_precision(sell_price, symbol_info['tick_size'])
                                            
                                        # Log the order details for debugging
                                        st.info(f"Attempting to place order: Quantity={quantity} (Original: {available_quantity}), Price={formatted_price}")
                                        
                                        # Place only the take profit limit order
                                        sell_order = client.create_order(
                                            symbol=position['symbol'],
                                            side=SIDE_SELL,
                                            type=ORDER_TYPE_LIMIT,
                                            timeInForce=TIME_IN_FORCE_GTC,
                                            quantity=quantity,
                                            price=str(formatted_price)
                                        )
                                        
                                        # Calculate stop loss price (but don't place order)
                                        stop_loss_price = position['current_price'] * (1 - stop_loss_percent / 100)
                                        stop_loss_price = round_price_precision(stop_loss_price, symbol_info['tick_size'])
                                        
                                        st.session_state.orders_config[position['symbol']] = {
                                            'sell_price': formatted_price,
                                            'stop_loss_price': stop_loss_price,
                                            'sell_order_id': sell_order['orderId'],
                                            'position_quantity': quantity,
                                            'order_type': 'TAKE_PROFIT'  # Track order type
                                        }
                                        st.success(f"Take profit order placed for {position['symbol']} at {formatted_price}")
                                        st.info(f"Stop loss will trigger at {stop_loss_price} (actively monitoring)")
                                except ValueError as ve:
                                    st.error(f"Order validation error: {str(ve)}")
                                except Exception as e:
                                    st.error(f"Error placing sell order: {str(e)}")
                                    
                                    # Show detailed symbol information for debugging
                                    if symbol_info and symbol_info.get('lot_filter'):
                                        lot_filter = symbol_info['lot_filter']
                                        st.error(f"LOT_SIZE filter - Min: {lot_filter['minQty']}, Max: {lot_filter['maxQty']}, Step: {lot_filter['stepSize']}")
                        
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