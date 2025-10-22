"""
Strategy Template - Copy this file to create new strategies

This template shows how to create custom strategies for the stock screener.
Follow the examples below and add your strategy to the StrategyManager.
"""

from stock_screener.strategy import Strategy, StrategyType, StrategyCategory
from typing import Dict, Any
import pandas as pd
import ta

class MyCustomStrategy(Strategy):
    """
    Template for creating a custom strategy
    
    Replace this with your strategy logic
    """
    
    def __init__(self, my_parameter: float = 1.0):
        super().__init__(
            name="My Custom Strategy",                    # Strategy display name
            description="Description of what this strategy does",  # Brief description
            strategy_type=StrategyType.FILTER,           # FILTER, RANKING, or SCORING
            category=StrategyCategory.TECHNICAL,         # Category for UI organization
            parameters={"my_parameter": my_parameter},   # Parameters that can be configured
            min_data_points=20,                          # Minimum days of data needed
            tags=["custom", "example"]                   # Tags for searching/filtering
        )
        self.my_parameter = my_parameter
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Define UI configuration for parameters
        This creates interactive controls in Streamlit
        """
        return {
            "my_parameter": {
                "type": "slider",           # slider, selectbox, number_input
                "min": 0.1,
                "max": 5.0,
                "default": 1.0,
                "step": 0.1,
                "label": "My Parameter",
                "help": "Description of what this parameter does"
            }
        }
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """
        Main strategy logic
        
        Args:
            stock_data: Dictionary containing 'quote' and 'historical' data
            
        Returns:
            bool: True if stock passes the strategy criteria
        """
        try:
            # Get data
            historical = stock_data.get('historical')
            quote = stock_data.get('quote')
            
            if not historical or not quote:
                return False
            
            # Convert to DataFrame if needed
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            # Check minimum data requirement
            if len(df) < self.min_data_points:
                return False
            
            # Prepare price data
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            df['high'] = pd.to_numeric(df['CH_TRADE_HIGH_PRICE'], errors='coerce')
            df['low'] = pd.to_numeric(df['CH_TRADE_LOW_PRICE'], errors='coerce')
            df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 10:
                return False
            
            # Example: Check if current price is above 20-day SMA
            sma_20 = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            current_price = float(quote.get('lastPrice', 0))
            
            if pd.isna(sma_20.iloc[-1]):
                return False
            
            # Your strategy logic here
            return current_price > (sma_20.iloc[-1] * self.my_parameter)
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error in MyCustomStrategy for {stock_data.get('symbol', 'Unknown')}: {e}")
            return False

# Example: More complex strategy with multiple conditions
class AdvancedMomentumStrategy(Strategy):
    """Example of a more complex strategy"""
    
    def __init__(self, rsi_threshold: float = 60, volume_multiplier: float = 1.5):
        super().__init__(
            name=f"Advanced Momentum (RSI>{rsi_threshold}, Vol>{volume_multiplier}x)",
            description="Combines RSI momentum with high volume",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.MOMENTUM,
            parameters={
                "rsi_threshold": rsi_threshold,
                "volume_multiplier": volume_multiplier
            },
            min_data_points=30,
            tags=["momentum", "rsi", "volume", "advanced"]
        )
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "rsi_threshold": {
                "type": "slider",
                "min": 50,
                "max": 80,
                "default": 60,
                "step": 5,
                "label": "RSI Threshold",
                "help": "Minimum RSI value for momentum"
            },
            "volume_multiplier": {
                "type": "slider",
                "min": 1.0,
                "max": 3.0,
                "default": 1.5,
                "step": 0.1,
                "label": "Volume Multiplier",
                "help": "Volume must be this many times above average"
            }
        }
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        try:
            historical = stock_data.get('historical')
            quote = stock_data.get('quote')
            
            if not historical or not quote:
                return False
            
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            if len(df) < 30:
                return False
            
            # Prepare data
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
            df = df.dropna()
            
            if len(df) < 20:
                return False
            
            # Condition 1: RSI above threshold
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi) or current_rsi < self.rsi_threshold:
                return False
            
            # Condition 2: Volume above average
            avg_volume = df['volume'].tail(20).mean()
            current_volume = float(quote.get('totalTradedVolume', 0))
            
            if current_volume < (avg_volume * self.volume_multiplier):
                return False
            
            # Condition 3: Price above 20-day SMA
            sma_20 = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            current_price = float(quote.get('lastPrice', 0))
            
            if pd.isna(sma_20.iloc[-1]) or current_price < sma_20.iloc[-1]:
                return False
            
            return True
            
        except (ValueError, TypeError, KeyError):
            return False

# How to add your strategy to the screener:
"""
1. Copy this template file
2. Modify the strategy class with your logic
3. Add your strategy to StrategyManager in strategy.py:

   def _register_default_strategies(self):
       # ... existing strategies ...
       self.register_strategy('my_custom', MyCustomStrategy())

4. Your strategy will automatically appear in the Streamlit UI!

# Strategy Chaining Examples:
"""
# Example 1: Programmatic chaining
builder = strategy_manager.create_chain_builder()
composite = (builder
    .add_strategy('my_custom', my_parameter=2.0)
    .add_strategy('rsi_oversold', LogicalOperator.AND, rsi_lower=60)
    .build("My Custom + RSI"))

# Example 2: Using configuration
configs = [
    {'key': 'my_custom', 'parameters': {'my_parameter': 1.5}},
    {'key': 'high_volume', 'operator': 'AND', 'parameters': {'volume_multiplier': 2.0}}
]
composite = strategy_manager.create_composite_strategy("My Chain", configs)

# Example 3: In Streamlit UI
# Just use "Chain Strategies" mode and add strategies interactively!
"""

# Available technical indicators from TA library:
"""
Trend Indicators:
- ta.trend.SMAIndicator (Simple Moving Average)
- ta.trend.EMAIndicator (Exponential Moving Average)
- ta.trend.MACD (MACD)
- ta.trend.BollingerBands (Bollinger Bands)

Momentum Indicators:
- ta.momentum.RSIIndicator (RSI)
- ta.momentum.StochasticOscillator (Stochastic)
- ta.momentum.WilliamsRIndicator (Williams %R)
- ta.momentum.ROCIndicator (Rate of Change)

Volume Indicators:
- ta.volume.VolumeSMAIndicator (Volume SMA)
- ta.volume.OnBalanceVolumeIndicator (OBV)

Volatility Indicators:
- ta.volatility.AverageTrueRange (ATR)
- ta.volatility.BollingerBands (Bollinger Bands)
"""