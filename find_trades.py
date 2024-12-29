import logging
from pathlib import Path
from config import Config
from services.trade_analyzer import TradeAnalyzer
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client
from utils.market_analysis import print_market_analysis

def find_trades():
    """
    Main function to analyze market conditions and find trading opportunities
    """
    print("\n=== Crypto Trading Signal Analysis ===")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = Config()
    
    # Verify environment setup
    if not Path('.env').exists():
        logger.error("❌ .env file not found. Please create one with Binance API credentials")
        return
        
    try:
        client = setup_binance_client()
        print("✅ Successfully connected to Binance API")
    except Exception as e:
        logger.error(f"❌ Error setting up Binance client: {str(e)}")
        return
        
    # Initialize trade analyzer
    analyzer = TradeAnalyzer(client, config)
    
    if not analyzer.initialize_traders():
        logger.error("❌ Failed to initialize traders")
        return
        
    # Generate and analyze trading signals
    try:
        signals = analyzer.generate_trading_signals()
        
        analyzed_pairs = 0
        found_signals = 0
        
        for symbol, signal in signals.items():
            if signal is None:
                logger.warning(f"No signal data available for {symbol}")
                continue
                
            analyzed_pairs += 1
            # Access the dataclass attributes directly
            market_conditions = signal.market_conditions
            
            if market_conditions:
                print_market_analysis(symbol, market_conditions)
                
                if signal.prediction > config.CONFIDENCE_THRESHOLD:
                    found_signals += 1
                    analyzer.print_trade_opportunity(symbol, signal)
                    
        print(f"\n=== Trading Summary ===")
        print(f"Total Pairs Analyzed: {analyzed_pairs}")
        print(f"Strong Signals Found: {found_signals}")
        
    except Exception as e:
        logger.error(f"Error during market analysis: {str(e)}")
        raise
        
if __name__ == "__main__":
    find_trades()