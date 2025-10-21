"""
Strategy module for stock screening strategies
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class Strategy(ABC):
    """Abstract base class for screening strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """Apply strategy to stock data and return True if stock passes"""
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get strategy information"""
        return {
            'name': self.name,
            'description': self.description
        }

class AboveVWAPStrategy(Strategy):
    """Strategy to find stocks trading above VWAP"""
    
    def __init__(self):
        super().__init__(
            name="Above VWAP",
            description="Stocks trading above Volume Weighted Average Price"
        )
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """Check if stock is trading above VWAP"""
        try:
            quote = stock_data.get('quote')
            if not quote:
                return False
            
            current_price = float(quote.get('lastPrice', 0))
            vwap = float(quote.get('averagePrice', 0))
            
            return current_price > vwap
        except (ValueError, TypeError):
            return False

class Near50EMAStrategy(Strategy):
    """Strategy to find stocks within 5% of 50-day EMA"""
    
    def __init__(self, tolerance_percent: float = 5.0):
        super().__init__(
            name=f"Near 50-day EMA ({tolerance_percent}%)",
            description=f"Stocks within {tolerance_percent}% of 50-day Exponential Moving Average"
        )
        self.tolerance_percent = tolerance_percent    

    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """Check if stock is within tolerance of 50-day EMA"""
        try:
            historical = stock_data.get('historical')
            quote = stock_data.get('quote')
            
            if not historical or not quote:
                return False
            
            # Convert to DataFrame if it's a list
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            if len(df) < 50:
                return False
            
            # Calculate 50-day EMA
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            
            current_price = float(quote.get('lastPrice', 0))
            
            # Check if within tolerance
            tolerance = (self.tolerance_percent / 100) * ema_50
            return abs(current_price - ema_50) <= tolerance
            
        except (ValueError, TypeError, KeyError):
            return False

class StrategyManager:
    """Manager class for handling multiple strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies"""
        self.register_strategy('above_vwap', AboveVWAPStrategy())
        self.register_strategy('near_50_ema', Near50EMAStrategy())
    
    def register_strategy(self, key: str, strategy: Strategy):
        """Register a new strategy"""
        self.strategies[key] = strategy
    
    def get_strategy(self, key: str) -> Strategy:
        """Get strategy by key"""
        return self.strategies.get(key)
    
    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get all available strategies with their info"""
        return {key: strategy.get_info() for key, strategy in self.strategies.items()}
    
    def run_strategy(self, strategy_key: str, stocks_data: List[Dict[str, Any]]) -> List[str]:
        """Run strategy on list of stock data and return passing stocks"""
        strategy = self.get_strategy(strategy_key)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_key}' not found")
        
        passing_stocks = []
        for stock_data in stocks_data:
            if strategy.apply(stock_data):
                passing_stocks.append(stock_data['symbol'])
        
        return passing_stocks