"""
Stock Screener Package
"""
from stock_screener.screener import StockScreener
from stock_screener.data_sources import NSEDataProvider, YFinanceDataProvider, DataManager
from stock_screener.strategy import StrategyManager, Strategy

__version__ = "0.1.0"
__all__ = ["StockScreener", "NSEDataProvider", "StrategyManager", "Strategy"]

def main() -> None:
    """Entry point for the stock screener CLI"""
    print("Stock Screener - Use 'uv run streamlit run src/stock_screener/app.py' to start the web interface")
