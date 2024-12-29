from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class MarketCondition:
    value: float
    threshold: float
    met: bool
    description: str

class MarketAnalyzer:
    """Utility class for analyzing market conditions"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_market_conditions(self, data) -> Dict[str, MarketCondition]:
        """Analyze current market conditions based on latest data"""
        if data.empty:
            raise ValueError("No data available for market analysis")
            
        latest = data.iloc[-1]
        
        return {
            'Volume Momentum': MarketCondition(
                value=latest['volume_momentum'],
                threshold=self.config.VOLUME_MOMENTUM_THRESHOLD,
                met=latest['volume_momentum'] > self.config.VOLUME_MOMENTUM_THRESHOLD,
                description='Trading volume compared to 24h average'
            ),
            'RSI': MarketCondition(
                value=latest['rsi'],
                threshold=self.config.RSI_THRESHOLD,
                met=latest['rsi'] < self.config.RSI_THRESHOLD,
                description='Relative Strength Index (not overbought)'
            ),
            'Bollinger Position': MarketCondition(
                value=latest['bollinger_position'],
                threshold=self.config.BOLLINGER_POSITION_THRESHOLD,
                met=latest['bollinger_position'] < self.config.BOLLINGER_POSITION_THRESHOLD,
                description='Position within Bollinger Bands'
            ),
            'VWAP Deviation': MarketCondition(
                value=latest['vwap_deviation'],
                threshold=self.config.VWAP_DEVIATION_THRESHOLD,
                met=abs(latest['vwap_deviation']) < self.config.VWAP_DEVIATION_THRESHOLD,
                description='Price deviation from VWAP'
            ),
            'Volatility': MarketCondition(
                value=latest['volatility_15m'],
                threshold=self.config.VOLATILITY_THRESHOLD,
                met=latest['volatility_15m'] < self.config.VOLATILITY_THRESHOLD,
                description='15-minute volatility'
            )
        }

def print_market_analysis(symbol: str, conditions: Dict[str, MarketCondition]):
    """Print formatted market analysis results"""
    print(f"\n{symbol} Market Analysis:")
    print("=" * 50)
    
    conditions_met = sum(1 for cond in conditions.values() if cond.met)
    print(f"Conditions Met: {conditions_met}/{len(conditions)}")
    
    print("\nDetailed Conditions:")
    print("-" * 50)
    for name, condition in conditions.items():
        status = "✅" if condition.met else "❌"
        print(f"{name}:")
        print(f"  Current: {condition.value:.4f}")
        print(f"  Threshold: {condition.threshold}")
        print(f"  Status: {status}")
        print(f"  Description: {condition.description}")
        print()