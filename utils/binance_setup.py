import os
from binance.client import Client
from dotenv import load_dotenv

def setup_binance_client() -> Client:
    """Initialize and return Binance API client"""
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Binance API credentials not found in .env file")
        
    return Client(api_key, api_secret)