from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

@dataclass
class Config:
    """Configuration settings for the trading system"""
    
    # Paths
    DATA_DIR: Path = Path('./data/symbols/')
    MODEL_DIR: Path = Path('./train_models')
    
    # Dates
    START_DATE: str = '2024-01-01'
    END_DATE: str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Model parameters
    EPOCHS: int = 500
    BATCH_SIZE: int = 256
    MIN_EPOCHS: int = 100
    PATIENCE: int = 50
    CONFIDENCE_THRESHOLD: float = 0.8
    
    # Trading parameters
    ENTRY_PRICE_FACTOR: float = 0.995  # 0.5% below current price
    TARGET_PRICE_FACTOR: float = 1.015  # 1.5% above entry price
    STOP_LOSS_FACTOR: float = 0.99     # 1% below entry price
    
    # Market conditions thresholds
    VOLUME_MOMENTUM_THRESHOLD: float = 0.8
    RSI_THRESHOLD: float = 75
    BOLLINGER_POSITION_THRESHOLD: float = 0.85
    VWAP_DEVIATION_THRESHOLD: float = 0.02
    VOLATILITY_THRESHOLD: float = 0.03
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)