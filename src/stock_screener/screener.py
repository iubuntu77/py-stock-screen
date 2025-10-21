"""
Main screener module that orchestrates data fetching and strategy execution
"""
from typing import List, Dict, Any
from stock_screener.data_provider import NSEDataProvider
from stock_screener.strategy import StrategyManager

class StockScreener:
    """Main stock screener class"""
    
    def __init__(self):
        self.data_provider = NSEDataProvider()
        self.strategy_manager = StrategyManager()
    
    def get_available_universes(self) -> List[str]:
        """Get available stock universes"""
        return self.data_provider.get_available_universes()
    
    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get available screening strategies"""
        return self.strategy_manager.get_available_strategies()
    
    def screen_stocks(self, universe: str, strategy_key: str) -> Dict[str, Any]:
        """
        Screen stocks from universe using specified strategy
        
        Args:
            universe: Stock universe (nifty50, nifty500, fno)
            strategy_key: Strategy identifier
            
        Returns:
            Dictionary with results and metadata
        """
        try:
            # Get stocks from universe
            print(f"Fetching stocks from {universe} universe...")
            stock_symbols = self.data_provider.get_universe_stocks(universe)
            
            if not stock_symbols:
                return {
                    'success': False,
                    'error': f'No stocks found for universe: {universe}',
                    'passing_stocks': [],
                    'total_stocks': 0
                }
            
            print(f"Found {len(stock_symbols)} stocks. Fetching data...")
            
            # Fetch data for all stocks
            stocks_data = []
            for i, symbol in enumerate(stock_symbols):
                print(f"Processing {symbol} ({i+1}/{len(stock_symbols)})")
                stock_data = self.data_provider.get_stock_data(symbol)
                if stock_data['quote']:  # Only add if data was fetched successfully
                    stocks_data.append(stock_data)
            
            # Apply strategy
            print(f"Applying strategy: {strategy_key}")
            passing_stocks = self.strategy_manager.run_strategy(strategy_key, stocks_data)
            
            return {
                'success': True,
                'passing_stocks': passing_stocks,
                'total_stocks': len(stocks_data),
                'universe': universe,
                'strategy': strategy_key,
                'strategy_info': self.strategy_manager.get_strategy(strategy_key).get_info()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'passing_stocks': [],
                'total_stocks': 0
            }