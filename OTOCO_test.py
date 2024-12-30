import logging
import time
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import os
from dotenv import load_dotenv

from binance.client import Client
from binance.enums import (
    SIDE_BUY, 
    SIDE_SELL, 
    ORDER_TYPE_MARKET,
    TIME_IN_FORCE_GTC
)
from binance.exceptions import BinanceAPIException

# -----------------------------------------------------------------------------
# Load your API keys from the .env file
# -----------------------------------------------------------------------------
load_dotenv()  # Reads .env in current dir
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# -----------------------------------------------------------------------------
# Setup Logging (optional)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class NeoTradeSimulator:
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.symbol = "NEOUSDT"  # Trading pair

    def get_symbol_info(self):
        """Get trading rules (filters) for the symbol."""
        return self.client.get_symbol_info(self.symbol)

    def get_current_price(self):
        """Get current symbol price."""
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        return float(ticker['price'])

    def calculate_trade_parameters(self, usdt_amount: float):
        """Compute trade parameters for entry, target, and stop prices."""
        symbol_info = self.get_symbol_info()
        lot_size = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')

        step_size = float(lot_size['stepSize'])
        min_qty = float(lot_size['minQty'])

        tick_size = float(price_filter['tickSize'])
        price_precision = len(str(tick_size).rstrip('0').split('.')[-1])

        current_price = self.get_current_price()
        entry_price = current_price
        target_price = round(entry_price * 1.015, price_precision)
        stop_price   = round(entry_price * 0.99,  price_precision)

        raw_quantity = usdt_amount / entry_price
        quantity = float(Decimal(str(raw_quantity)).quantize(
            Decimal(str(step_size)),
            rounding=ROUND_DOWN
        ))

        if quantity < min_qty:
            raise ValueError(f"Calculated quantity {quantity} is below the min lot size {min_qty}")

        return {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "quantity": quantity
        }

    def place_otoco_order(self, usdt_amount: float = 15.0):
        """Simulate an OTOCO order."""
        try:
            params = self.calculate_trade_parameters(usdt_amount)
            response = self.client.post(
                "/api/v3/orderList/otoco",
                data={
                    "symbol": self.symbol,
                    "workingType": "LIMIT",
                    "workingSide": SIDE_BUY,
                    "workingPrice": str(params["entry_price"]),
                    "workingQuantity": str(params["quantity"]),
                    "pendingSide": SIDE_SELL,
                    "pendingQuantity": str(params["quantity"]),
                    "pendingAboveType": "TAKE_PROFIT_LIMIT",
                    "pendingAbovePrice": str(params["target_price"]),
                    "pendingAboveStopPrice": str(params["target_price"]),
                    "pendingAboveTimeInForce": TIME_IN_FORCE_GTC,
                    "pendingBelowType": "STOP_LOSS_LIMIT",
                    "pendingBelowPrice": str(params["stop_price"]),
                    "pendingBelowStopPrice": str(params["stop_price"]),
                    "pendingBelowTimeInForce": TIME_IN_FORCE_GTC,
                    "recvWindow": 60000,
                    "timestamp": int(time.time() * 1000),
                }
            )
            return response.json()

        except BinanceAPIException as e:
            self.logger.error(f"Binance API Error: {e}")
        except Exception as e:
            self.logger.error(f"Error: {e}")

def main():
    client = Client(api_key=API_KEY, api_secret=API_SECRET)
    simulator = NeoTradeSimulator(client)
    usdt_amount = float(input("\nEnter USDT amount for trade: "))
    simulator.place_otoco_order(usdt_amount)

if __name__ == "__main__":
    main()
