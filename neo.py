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

# -----------------------------------------------------------------------------
# Class: NeoTradeSimulator
# -----------------------------------------------------------------------------
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
        """
        1) Figure out how many coins we can buy with the given USDT amount.
        2) Compute a target price (+1.5%) and stop price (-1%).
        3) Format them according to the symbol's LOT_SIZE and PRICE_FILTER.
        """
        symbol_info = self.get_symbol_info()

        # 1) Extract filters
        lot_size = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')

        step_size = float(lot_size['stepSize'])
        min_qty = float(lot_size['minQty'])

        tick_size = float(price_filter['tickSize'])
        price_precision = len(str(tick_size).rstrip('0').split('.')[-1])

        # 2) Current Price
        current_price = self.get_current_price()

        # 3) Desired target & stop
        entry_price = current_price
        target_price = round(entry_price * 1.015, price_precision)  # +1.5%
        stop_price   = round(entry_price * 0.99,  price_precision)   # -1%

        # 4) Calculate quantity (round down to step_size)
        raw_quantity = usdt_amount / entry_price
        quantity = float(Decimal(str(raw_quantity)).quantize(
            Decimal(str(step_size)),
            rounding=ROUND_DOWN
        ))

        if quantity < min_qty:
            raise ValueError(
                f"Calculated quantity {quantity} is below the min lot size {min_qty}"
            )

        return {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "quantity": quantity
        }

    def simulate_trade(self, usdt_amount: float = 15.0):
        """
        1) Market BUY (to acquire the coin).
        2) OCO SELL (Stop-Loss & Take-Profit) to exit the position.
        """
        try:
            print("\n=== NEO Trade Simulation ===")
            current_price = self.get_current_price()
            print(f"Current NEO price: {current_price}")

            # 1) Compute trade details
            params = self.calculate_trade_parameters(usdt_amount)
            print("\nCalculated Trade Parameters:")
            print(f"  USDT Amount: {usdt_amount}")
            print(f"  Quantity: {params['quantity']}")
            print(f"  Entry Price (Market): {params['entry_price']}")
            print(f"  Take Profit (Limit): {params['target_price']}")
            print(f"  Stop Loss (Stop Price): {params['stop_price']}")

            confirm = input("\nPlace this trade? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Trade cancelled.")
                return

            # 2) Place MARKET BUY
            print("\nPlacing Market BUY order...")
            buy_order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=params['quantity']
            )
            print("Market BUY order placed successfully!")
            print(f"  Order ID: {buy_order['orderId']}")

            # (Optional) Wait for fill if necessary. 
            # Typically for small trades, it should fill instantly.

            # 3) Place OCO SELL (TP + SL)
            print("\nPlacing OCO SELL order...")
            oco_order = self.client.create_oco_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                quantity=params['quantity'],
                price=str(params['target_price']),         # Limit (TP) price
                stopPrice=str(params['stop_price']),       # Stop trigger price
                stopLimitPrice=str(params['stop_price']),  # Stop limit
                stopLimitTimeInForce=TIME_IN_FORCE_GTC,
                # The key param that avoids 'aboveType' error:
                stopLimitType='STOP_LOSS_LIMIT',           
                # If you still get 'aboveType' error, also try:
                # priceProtect='FALSE',
            )

            print("OCO SELL order placed successfully!")
            print(f"  Order List ID: {oco_order['orderListId']}")
            print(f"  TP Limit Price: {params['target_price']}")
            print(f"  SL Trigger/Limit Price: {params['stop_price']}")

            return buy_order, oco_order

        except BinanceAPIException as e:
            print(f"\nBinance API Error: {e.message}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
        except Exception as e:
            print(f"\nError: {str(e)}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # 1) Create client from .env keys
    client = Client(api_key=API_KEY, api_secret=API_SECRET)
    
    # 2) Check USDT balance
    account_info = client.get_account()
    balances = {bal['asset']: float(bal['free']) for bal in account_info['balances']}
    usdt_balance = balances.get("USDT", 0.0)
    print(f"\nAvailable USDT: {usdt_balance}")

    # 3) Ask user for trade size
    try:
        amount = float(input("\nEnter USDT amount for trade: "))
        if amount <= 0 or amount > usdt_balance:
            print("Invalid amount.")
            return
    except ValueError:
        print("Invalid input. Must be a number.")
        return

    # 4) Run the trade simulation (Buy + OCO)
    simulator = NeoTradeSimulator(client)
    simulator.simulate_trade(amount)

if __name__ == "__main__":
    main()
