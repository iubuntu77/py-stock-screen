"""
Consolidated data sources module containing all data providers and manager
"""
import os
import time
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# NSE Python imports with fallback
try:
    from nsepython import nsefetch, equity_history
except ImportError:
    def nsefetch(symbol):
        return None
    def equity_history(symbol, segment, start_date, end_date):
        return None

# Mock data functions for testing
def nifty_list() -> List[str]:
    """Mock NIFTY 50 stocks"""
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR"]

def nifty500_list() -> List[str]:
    """Mock NIFTY 500 stocks"""
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "BAJFINANCE", "ASIANPAINT"]

def fnolist() -> List[str]:
    """Mock F&O stocks"""
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
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the data provider is working"""
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
            quote_data = nsefetch(symbol)
            historical_data = equity_history(symbol, "EQ", start_date="01-01-2024", end_date="31-12-2024")
            
            if not quote_data:
                quote_data = self._generate_mock_quote_data(symbol)
            
            if not historical_data:
                historical_data = self._generate_mock_historical_data(symbol)
            
            return {
                'symbol': symbol,
                'quote': quote_data,
                'historical': historical_data,
                'source': 'nse'
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}, using mock data: {e}")
            return {
                'symbol': symbol,
                'quote': self._generate_mock_quote_data(symbol),
                'historical': self._generate_mock_historical_data(symbol),
                'source': 'nse_mock'
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
        data = []
        base_price = random.uniform(100, 3000)
        
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
            
            base_price = close_price * random.uniform(0.98, 1.02)
        
        return data
    
    def test_connection(self) -> bool:
        """Test NSE connection"""
        try:
            result = nsefetch("RELIANCE")
            return result is not None
        except Exception:
            return False

class YFinanceDataProvider(DataProvider):
    """YFinance data provider for Indian stocks"""
    
    NSE_SUFFIX = ".NS"
    
    # Comprehensive stock lists
    NIFTY_50_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", 
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT", 
        "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN", "WIPRO",
        "NESTLEIND", "BAJFINANCE", "HCLTECH", "POWERGRID", "NTPC", "TATAMOTORS",
        "COALINDIA", "BAJAJFINSV", "M&M", "TECHM", "INDUSINDBK", "ADANIENT",
        "ONGC", "TATASTEEL", "CIPLA", "APOLLOHOSP", "DRREDDY", "EICHERMOT",
        "JSWSTEEL", "BRITANNIA", "GRASIM", "BPCL", "DIVISLAB", "HEROMOTOCO",
        "TATACONSUM", "BAJAJ-AUTO", "HINDALCO", "ADANIPORTS", "UPL", "SBILIFE",
        "HDFCLIFE", "LTIM"
    ]
    
    NIFTY_500_SYMBOLS = NIFTY_50_SYMBOLS + [
        "GODREJCP", "PIDILITIND", "BERGEPAINT", "MARICO", "DABUR", "COLPAL",
        "MCDOWELL-N", "UBL", "AMBUJACEM", "ACC", "SHREECEM", "RAMCOCEM",
        "JINDALSTEL", "SAIL", "NMDC", "VEDL", "HINDZINC", "NATIONALUM",
        "BANKBARODA", "PNB", "CANBK", "IOC", "GAIL", "PETRONET", "MOTHERSON",
        "BOSCHLTD", "ESCORTS", "BAJAJHLDNG", "SIEMENS", "ABB", "HAVELLS"
    ]
    
    FNO_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", "SBIN",
        "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK", "MARUTI",
        "SUNPHARMA", "ULTRACEMCO", "TITAN", "WIPRO", "NESTLEIND", "BAJFINANCE",
        "HCLTECH", "POWERGRID", "NTPC", "TATAMOTORS", "COALINDIA", "BAJAJFINSV",
        "M&M", "TECHM", "INDUSINDBK", "ONGC", "TATASTEEL", "CIPLA", "APOLLOHOSP",
        "DRREDDY", "EICHERMOT", "JSWSTEEL", "BRITANNIA", "GRASIM", "BPCL",
        "DIVISLAB", "HEROMOTOCO", "BAJAJ-AUTO", "HINDALCO", "ADANIPORTS", "UPL"
    ]
    
    def __init__(self):
        self.timeout = int(os.getenv('REQUEST_TIMEOUT', 30))
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', 3))
        self.universe_mapping = {
            'nifty50': 'NIFTY 50',
            'nifty500': 'NIFTY 500', 
            'fno': 'F&O'
        }
    
    def _get_yf_symbol(self, nse_symbol: str) -> str:
        """Convert NSE symbol to YFinance format"""
        return f"{nse_symbol}{self.NSE_SUFFIX}"
    
    def get_universe_stocks(self, universe: str) -> List[str]:
        """Get stocks for specified universe"""
        try:
            if universe == 'nifty50':
                return self.NIFTY_50_SYMBOLS.copy()
            elif universe == 'nifty500':
                return self.NIFTY_500_SYMBOLS.copy()
            elif universe == 'fno':
                return self.FNO_SYMBOLS.copy()
            else:
                raise ValueError(f"Unknown universe: {universe}")
        except Exception as e:
            print(f"Error fetching universe {universe}: {e}")
            return []
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock data using yfinance"""
        yf_symbol = self._get_yf_symbol(symbol)
        
        for attempt in range(self.retry_attempts):
            try:
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                hist_data = ticker.history(start=start_date, end=end_date, interval="1d")
                
                if hist_data.empty:
                    raise ValueError(f"No data available for {symbol}")
                
                current_price = hist_data['Close'].iloc[-1]
                prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
                
                quote_data = {
                    'symbol': symbol,
                    'lastPrice': float(current_price),
                    'averagePrice': float(hist_data['Close'].tail(20).mean()) if len(hist_data) >= 20 else float(current_price),
                    'change': float(current_price - prev_close),
                    'pChange': float(((current_price - prev_close) / prev_close) * 100) if prev_close != 0 else 0,
                    'totalTradedVolume': int(hist_data['Volume'].iloc[-1]),
                    'marketCap': info.get('marketCap', 0),
                    'pe': info.get('trailingPE', 0),
                    'pb': info.get('priceToBook', 0)
                }
                
                historical_data = []
                for date, row in hist_data.iterrows():
                    historical_data.append({
                        'CH_TIMESTAMP': date.strftime('%d-%b-%Y'),
                        'CH_CLOSING_PRICE': float(row['Close']),
                        'CH_TRADE_HIGH_PRICE': float(row['High']),
                        'CH_TRADE_LOW_PRICE': float(row['Low']),
                        'CH_OPENING_PRICE': float(row['Open']),
                        'CH_TOT_TRADED_QTY': int(row['Volume'])
                    })
                
                return {
                    'symbol': symbol,
                    'quote': quote_data,
                    'historical': historical_data,
                    'source': 'yfinance'
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)
                else:
                    return self._generate_fallback_data(symbol)
    
    def _generate_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback data when yfinance fails"""
        import random
        
        base_price = random.uniform(100, 3000)
        
        quote_data = {
            'symbol': symbol,
            'lastPrice': base_price,
            'averagePrice': base_price * random.uniform(0.95, 1.05),
            'change': random.uniform(-50, 50),
            'pChange': random.uniform(-5, 5),
            'totalTradedVolume': random.randint(100000, 10000000),
            'marketCap': random.randint(10000, 1000000) * 10000000,
            'pe': random.uniform(10, 50),
            'pb': random.uniform(1, 10)
        }
        
        historical_data = []
        current_price = base_price
        
        for i in range(60):
            date = datetime.now() - timedelta(days=60-i)
            price_variation = random.uniform(0.95, 1.05)
            close_price = current_price * price_variation
            
            historical_data.append({
                'CH_TIMESTAMP': date.strftime('%d-%b-%Y'),
                'CH_CLOSING_PRICE': close_price,
                'CH_TRADE_HIGH_PRICE': close_price * random.uniform(1.0, 1.05),
                'CH_TRADE_LOW_PRICE': close_price * random.uniform(0.95, 1.0),
                'CH_OPENING_PRICE': close_price * random.uniform(0.98, 1.02),
                'CH_TOT_TRADED_QTY': random.randint(10000, 1000000)
            })
            
            current_price = close_price * random.uniform(0.98, 1.02)
        
        return {
            'symbol': symbol,
            'quote': quote_data,
            'historical': historical_data,
            'source': 'yfinance_fallback'
        }
    
    def test_connection(self) -> bool:
        """Test if yfinance is working"""
        try:
            ticker = yf.Ticker("RELIANCE.NS")
            info = ticker.info
            return bool(info and len(info) > 5)
        except Exception:
            return False

class DataManager:
    """Manages multiple data providers with automatic fallback"""
    
    def __init__(self):
        """Initialize data manager with configured providers"""
        self.providers = {}
        self.current_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available data providers"""
        # Initialize YFinance provider
        try:
            self.providers['yfinance'] = YFinanceDataProvider()
        except Exception as e:
            print(f"Failed to initialize YFinance provider: {e}")
        
        # Initialize NSE provider
        try:
            self.providers['nse'] = NSEDataProvider()
        except Exception as e:
            print(f"Failed to initialize NSE provider: {e}")
        
        # Set current provider based on env config
        default_provider = os.getenv('DEFAULT_DATA_PROVIDER', 'yfinance')
        if default_provider in self.providers:
            self.current_provider = self.providers[default_provider]
        else:
            if self.providers:
                self.current_provider = list(self.providers.values())[0]
                print(f"Default provider '{default_provider}' not available, using fallback")
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            try:
                is_working = provider.test_connection()
                status[name] = {
                    'available': True,
                    'working': is_working,
                    'type': type(provider).__name__
                }
            except Exception as e:
                status[name] = {
                    'available': False,
                    'working': False,
                    'error': str(e),
                    'type': type(provider).__name__
                }
        
        return status
    
    def set_provider(self, provider_name: str) -> bool:
        """Set active data provider"""
        if provider_name in self.providers:
            self.current_provider = self.providers[provider_name]
            return True
        return False
    
    def get_current_provider_name(self) -> str:
        """Get name of current provider"""
        for name, provider in self.providers.items():
            if provider == self.current_provider:
                return name
        return "unknown"
    
    def get_universe_stocks(self, universe: str) -> List[str]:
        """Get stocks for specified universe with fallback"""
        if not self.current_provider:
            raise RuntimeError("No data provider available")
        
        try:
            return self.current_provider.get_universe_stocks(universe)
        except Exception as e:
            print(f"Primary provider failed: {e}")
            return self._try_fallback_provider('get_universe_stocks', universe)
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data with fallback support"""
        if not self.current_provider:
            raise RuntimeError("No data provider available")
        
        try:
            data = self.current_provider.get_stock_data(symbol)
            data['data_source'] = self.get_current_provider_name()
            return data
        except Exception as e:
            print(f"Primary provider failed for {symbol}: {e}")
            return self._try_fallback_provider('get_stock_data', symbol)
    
    def get_available_universes(self) -> List[str]:
        """Get available universes"""
        return ['nifty50', 'nifty500', 'fno']
    
    def _try_fallback_provider(self, method_name: str, *args) -> Any:
        """Try fallback provider for the given method"""
        fallback_name = os.getenv('FALLBACK_DATA_PROVIDER', 'nse')
        
        if fallback_name in self.providers and fallback_name != self.get_current_provider_name():
            try:
                fallback_provider = self.providers[fallback_name]
                method = getattr(fallback_provider, method_name)
                result = method(*args)
                
                if method_name == 'get_stock_data' and isinstance(result, dict):
                    result['data_source'] = f"{fallback_name}_fallback"
                
                print(f"Fallback provider '{fallback_name}' succeeded")
                return result
            except Exception as e:
                print(f"Fallback provider '{fallback_name}' also failed: {e}")
        
        raise RuntimeError(f"All data providers failed for {method_name}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current data setup"""
        return {
            'current_provider': self.get_current_provider_name(),
            'fallback_provider': os.getenv('FALLBACK_DATA_PROVIDER', 'nse'),
            'available_providers': list(self.providers.keys()),
            'provider_status': self.get_provider_status(),
            'config': {
                'timeout': int(os.getenv('REQUEST_TIMEOUT', 30)),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', 3))
            }
        }