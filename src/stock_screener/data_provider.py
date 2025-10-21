"""
Data provider module for fetching stock data from NSE
"""
try:
    from nsepython import nsefetch, equity_history
except ImportError:
    # Mock functions if nsepython is not available
    def nsefetch(symbol):
        return None
    def equity_history(symbol, segment, start_date, end_date):
        return None

import pandas as pd
from typing import List, Dict, Any
from abc import ABC, abstractmethod

# Mock data functions for testing
def nifty_list() -> List[str]:
    """Mock NIFTY 50 stocks - returns 6 popular stocks"""
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR"]

def nifty500_list() -> List[str]:
    """Mock NIFTY 500 stocks - returns 8 stocks including mid-cap"""
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "BAJFINANCE", "ASIANPAINT"]

def fnolist() -> List[str]:
    """Mock F&O stocks - returns 7 stocks available for derivatives trading"""
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BAJFINANCE", "SBIN"]

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def get_universe_stocks(self, universe: str) -> List[str]:
        """Get list of stocks for a given universe"""
        pass
    
    @abstractmethod
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data for a given symbol"""
        pass

class NSEDataProvider(DataProvider):
    """NSE data provider using nsepython"""
    
    def __init__(self):
        self.universe_mapping = {
            'nifty50': 'NIFTY 50',
            'nifty500': 'NIFTY 500', 
            'fno': 'F&O'
        }
    
    def get_universe_stocks(self, universe: str) -> List[str]:
        """Get stocks for specified universe"""
        try:
            if universe == 'nifty50':
                return nifty_list()
            elif universe == 'nifty500':
                return nifty500_list()
            elif universe == 'fno':
                return fnolist()
            else:
                raise ValueError(f"Unknown universe: {universe}")
        except Exception as e:
            print(f"Error fetching universe {universe}: {e}")
            return []
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock data"""
        try:
            # Try to get real data first
            quote_data = nsefetch(symbol)
            historical_data = equity_history(symbol, "EQ", start_date="01-01-2024", end_date="31-12-2024")
            
            # If real data fails, use mock data
            if not quote_data:
                quote_data = self._generate_mock_quote_data(symbol)
            
            if not historical_data:
                historical_data = self._generate_mock_historical_data(symbol)
            
            return {
                'symbol': symbol,
                'quote': quote_data,
                'historical': historical_data
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}, using mock data: {e}")
            return {
                'symbol': symbol,
                'quote': self._generate_mock_quote_data(symbol),
                'historical': self._generate_mock_historical_data(symbol)
            }
    
    def _generate_mock_quote_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock quote data for testing"""
        import random
        base_price = random.uniform(100, 3000)
        return {
            'lastPrice': base_price,
            'averagePrice': base_price * random.uniform(0.95, 1.05),
            'change': random.uniform(-50, 50),
            'pChange': random.uniform(-5, 5),
            'totalTradedVolume': random.randint(100000, 10000000),
            'symbol': symbol
        }
    
    def _generate_mock_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock historical data for testing"""
        import random
        from datetime import datetime, timedelta
        
        data = []
        base_price = random.uniform(100, 3000)
        
        # Generate 60 days of mock data
        for i in range(60):
            date = datetime.now() - timedelta(days=60-i)
            price_variation = random.uniform(0.95, 1.05)
            close_price = base_price * price_variation
            
            data.append({
                'CH_TIMESTAMP': date.strftime('%d-%b-%Y'),
                'CH_CLOSING_PRICE': close_price,
                'CH_TRADE_HIGH_PRICE': close_price * random.uniform(1.0, 1.05),
                'CH_TRADE_LOW_PRICE': close_price * random.uniform(0.95, 1.0),
                'CH_OPENING_PRICE': close_price * random.uniform(0.98, 1.02),
                'CH_TOT_TRADED_QTY': random.randint(10000, 1000000)
            })
            
            # Slight trend for next day
            base_price = close_price * random.uniform(0.98, 1.02)
        
        return data
    
    def get_available_universes(self) -> List[str]:
        """Get list of available stock universes"""
        return list(self.universe_mapping.keys())