import logging
import time
from decimal import Decimal, ROUND_DOWN
import os
from dotenv import load_dotenv
import requests
import hmac
import hashlib

from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load your API keys from the .env file
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class SimpleOTOCOTrader:
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.symbol = "NEOUSDT"

    def get_symbol_info(self):
        """Get trading rules (filters) for the symbol."""
        info = self.client.get_symbol_info(self.symbol)
        if not info:
            raise ValueError(f"No symbol info found for {self.symbol}")
        return info

    def get_market_price(self):
        """Get current market price."""
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        return float(ticker['price'])

    def calculate_order_prices(self, market_price: float, symbol_info: dict):
        """Calculate entry, target and stop prices."""
        price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
        tick_size = float(price_filter['tickSize'])
        decimals = len(str(tick_size).rstrip('0').split('.')[-1])

        # Entry slightly below market for limit buy
        entry_price = round(market_price * 0.998, decimals)  # -0.2% entry
        
        # Target above entry
        target_price = round(entry_price * 1.015, decimals)  # +1.5% from entry
        
        # Stop loss below entry price
        stop_price = round(entry_price * 0.995, decimals)   # -0.5% from entry price

        return entry_price, target_price, stop_price

        return entry_price, target_price, stop_price

    def calculate_quantity(self, price: float, usdt_amount: float, symbol_info: dict):
        """Calculate order quantity."""
        lot_size = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size['stepSize'])
        min_qty = float(lot_size['minQty'])

        raw_quantity = usdt_amount / price
        quantity = float(Decimal(str(raw_quantity)).quantize(
            Decimal(str(step_size)),
            rounding=ROUND_DOWN
        ))

        if quantity < min_qty:
            raise ValueError(f"Calculated quantity {quantity} is below minimum {min_qty}")

        return quantity

    def place_otoco_order(self, usdt_amount: float = 15.0):
        """Place OTOCO order with percentage-based prices."""
        try:
            # Get current market conditions
            symbol_info = self.get_symbol_info()
            market_price = self.get_market_price()
            
            # Calculate prices and quantity
            entry_price, target_price, stop_price = self.calculate_order_prices(market_price, symbol_info)
            quantity = self.calculate_quantity(entry_price, usdt_amount, symbol_info)

            self.logger.info(f"Market price: {market_price}")
            self.logger.info(f"Entry price: {entry_price}")
            self.logger.info(f"Target price: {target_price}")
            self.logger.info(f"Stop price: {stop_price}")
            self.logger.info(f"Quantity: {quantity}")

            # Build OTOCO order parameters
            otoco_params = {
                "symbol": self.symbol,
                "timestamp": str(int(time.time() * 1000)),
                
                # Order A: Limit Buy Entry
                "workingType": "LIMIT",
                "workingSide": "BUY",
                "workingTimeInForce": "GTC",
                "workingPrice": str(entry_price),
                "workingQuantity": str(quantity),
                
                # Pending Orders Direction
                "pendingSide": "SELL",
                "pendingQuantity": str(quantity),
                
                # Order B: Take Profit (Limit Maker)
                "pendingAboveType": "LIMIT_MAKER",
                "pendingAbovePrice": str(target_price),
                
                # Order C: Stop Loss (Stop Loss Market)
                "pendingBelowType": "STOP_LOSS",
                "pendingBelowStopPrice": str(stop_price),
                
                "recvWindow": "5000"
            }

            # Generate signature string
            sorted_params = dict(sorted(otoco_params.items()))
            query_string = '&'.join([f"{key}={value}" for key, value in sorted_params.items()])
            
            # Create signature
            signature = hmac.new(
                bytes(self.client.API_SECRET, 'utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to query string
            query_string = f"{query_string}&signature={signature}"

            # Send OTOCO request
            endpoint = 'https://api.binance.com/api/v3/orderList/otoco'
            headers = {
                'X-MBX-APIKEY': self.client.API_KEY,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            self.logger.info("Sending OTOCO request...")
            response = requests.post(f"{endpoint}?{query_string}", headers=headers)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(error_message)
                raise Exception(error_message)

            response_data = response.json()
            self.logger.info("OTOCO order placed successfully")
            return response_data

        except Exception as e:
            self.logger.error(f"Error placing OTOCO order: {e}")
            raise

            # Generate signature string
            sorted_params = dict(sorted(otoco_params.items()))
            query_string = '&'.join([f"{key}={value}" for key, value in sorted_params.items()])
            
            # Create signature
            signature = hmac.new(
                bytes(self.client.API_SECRET, 'utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to query string
            query_string = f"{query_string}&signature={signature}"

            # Send OTOCO request
            endpoint = 'https://api.binance.com/api/v3/orderList/otoco'
            headers = {
                'X-MBX-APIKEY': self.client.API_KEY,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            self.logger.info("Sending OTOCO request...")
            response = requests.post(f"{endpoint}?{query_string}", headers=headers)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(error_message)
                raise Exception(error_message)

            response_data = response.json()
            self.logger.info("OTOCO order placed successfully")
            return response_data

        except Exception as e:
            self.logger.error(f"Error placing OTOCO order: {e}")
            raise

            # Generate signature string
            sorted_params = dict(sorted(otoco_params.items()))
            query_string = '&'.join([f"{key}={value}" for key, value in sorted_params.items()])
            
            # Create signature
            signature = hmac.new(
                bytes(self.client.API_SECRET, 'utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to query string
            query_string = f"{query_string}&signature={signature}"

            # Send OTOCO request
            endpoint = 'https://api.binance.com/api/v3/orderList/otoco'
            headers = {
                'X-MBX-APIKEY': self.client.API_KEY,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            self.logger.info("Sending OTOCO request...")
            response = requests.post(f"{endpoint}?{query_string}", headers=headers)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(error_message)
                raise Exception(error_message)

            response_data = response.json()
            self.logger.info("OTOCO order placed successfully")
            return response_data

        except Exception as e:
            self.logger.error(f"Error placing OTOCO order: {e}")
            raise

def main():
    logging.info("Starting Simple OTOCO Trader...")
    
    try:
        client = Client(API_KEY, API_SECRET)
        trader = SimpleOTOCOTrader(client)
        
        # Test connection
        client.ping()
        logging.info("Connected to Binance")
        
        # Place OTOCO order
        usdt_amount = float(input("\nEnter USDT amount for trade: "))
        response = trader.place_otoco_order(usdt_amount)
        
        # Print order details
        print("\n=== OTOCO Order Details ===")
        print(f"Order List ID: {response['orderListId']}")
        print(f"Status: {response['listStatusType']} - {response['listOrderStatus']}")
        
        for report in response.get('orderReports', []):
            order_type = report['type']
            print(f"\n{'📥' if order_type == 'LIMIT' else '🎯' if 'TAKE_PROFIT' in order_type else '🛑'} {order_type} Order:")
            print(f"Order ID: {report['orderId']}")
            print(f"Side: {report['side']}")
            print(f"Price: {report['price']}")
            if 'stopPrice' in report:
                print(f"Stop Price: {report['stopPrice']}")
            print(f"Status: {report['status']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        logging.info("Completed")

if __name__ == "__main__":
    main()