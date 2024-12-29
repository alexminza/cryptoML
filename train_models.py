import logging
from pathlib import Path
from config import Config
from services.model_trainer import ModelTrainingService
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client

def train_models():
    """
    Main function to train all crypto trading models
    """
    print("\n=== Crypto Trading Model Training ===")
    
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
        
    # Initialize model trainer
    trainer = ModelTrainingService(client, config)
    
    if not trainer.initialize_traders():
        logger.error("❌ Failed to initialize traders")
        return
        
    # Train models
    print(f"\nTraining models for {len(trainer.symbols)} pairs...")
    trainer.train_all_models(config.START_DATE, config.END_DATE)
    print("\n✅ Training completed")

if __name__ == "__main__":
    train_models()