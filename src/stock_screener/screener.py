"""
Main screener module that orchestrates data fetching and strategy execution
"""
from typing import List, Dict, Any, Optional
import time
import hashlib
import json
import os
from datetime import datetime, timedelta
from stock_screener.data_sources import DataManager
from stock_screener.strategy import StrategyManager
from stock_screener.audit_logger import get_audit_logger

class StockScreener:
    """Main stock screener class with caching support"""
    
    def __init__(self, enable_cache: bool = None, cache_duration_hours: int = None):
        self.data_manager = DataManager()
        self.strategy_manager = StrategyManager()
        
        # Caching configuration from environment or defaults
        if enable_cache is None:
            enable_cache = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
        if cache_duration_hours is None:
            cache_duration_hours = int(os.getenv('CACHE_DURATION_HOURS', '4'))
        
        self.enable_cache = enable_cache
        self.cache_duration_hours = cache_duration_hours
        
        # In-memory cache for stock data
        self._memory_cache = {}
        
        # Cache directory for persistent storage
        self.cache_dir = os.path.join(os.getcwd(), '.cache', 'stock_data')
        if enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Audit logger for verification
        self.audit_logger = get_audit_logger()
    
    def get_available_universes(self) -> List[str]:
        """Get available stock universes"""
        return self.data_manager.get_available_universes()
    
    def get_data_provider_info(self) -> Dict[str, Any]:
        """Get information about data providers"""
        return self.data_manager.get_provider_info()
    
    def set_data_provider(self, provider_name: str) -> bool:
        """Set the active data provider"""
        return self.data_manager.set_provider(provider_name)
    
    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get available screening strategies"""
        return self.strategy_manager.get_available_strategies()
    
    def _get_cache_key(self, universe: str, provider: str) -> str:
        """Generate cache key for universe and provider combination"""
        return f"{universe}_{provider}_{datetime.now().strftime('%Y%m%d_%H')}"
    
    def _is_cache_valid(self, cache_timestamp: float) -> bool:
        """Check if cache is still valid based on timestamp"""
        if not self.enable_cache:
            return False
        
        cache_age_hours = (time.time() - cache_timestamp) / 3600
        return cache_age_hours < self.cache_duration_hours
    
    def _get_cached_data(self, universe: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached stock data for universe"""
        if not self.enable_cache:
            return None
        
        provider = self.data_manager.get_current_provider_name()
        cache_key = self._get_cache_key(universe, provider)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            cache_entry = self._memory_cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                print(f"📋 Using cached data from memory for {universe} ({len(cache_entry['data'])} stocks)")
                self.audit_logger.log_cache_operation("hit", universe, size=len(cache_entry['data']))
                return cache_entry['data']
            else:
                # Remove expired cache
                del self._memory_cache[cache_key]
        
        # Check persistent cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                
                if self._is_cache_valid(cache_entry['timestamp']):
                    print(f"💾 Using cached data from disk for {universe} ({len(cache_entry['data'])} stocks)")
                    # Load into memory cache for faster access
                    self._memory_cache[cache_key] = cache_entry
                    self.audit_logger.log_cache_operation("hit", universe, size=len(cache_entry['data']))
                    return cache_entry['data']
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
            except (json.JSONDecodeError, KeyError, OSError):
                # Remove corrupted cache file
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
        
        return None
    
    def _cache_data(self, universe: str, stocks_data: List[Dict[str, Any]]) -> None:
        """Cache stock data for universe"""
        if not self.enable_cache or not stocks_data:
            return
        
        provider = self.data_manager.get_current_provider_name()
        cache_key = self._get_cache_key(universe, provider)
        
        cache_entry = {
            'timestamp': time.time(),
            'universe': universe,
            'provider': provider,
            'data': stocks_data,
            'count': len(stocks_data)
        }
        
        # Store in memory cache
        self._memory_cache[cache_key] = cache_entry
        
        # Store in persistent cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, default=str)  # default=str handles datetime objects
            print(f"💾 Cached {len(stocks_data)} stocks for {universe}")
            self.audit_logger.log_cache_operation("store", universe, size=len(stocks_data))
        except (OSError, TypeError) as e:
            print(f"Warning: Could not save cache to disk: {e}")
    
    def clear_cache(self, universe: Optional[str] = None) -> None:
        """Clear cache for specific universe or all universes"""
        if universe:
            # Clear specific universe cache
            keys_to_remove = [key for key in self._memory_cache.keys() if key.startswith(f"{universe}_")]
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            # Clear disk cache
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(f"{universe}_"):
                        try:
                            os.remove(os.path.join(self.cache_dir, filename))
                        except OSError:
                            pass
            print(f"🗑️ Cleared cache for {universe}")
        else:
            # Clear all cache
            self._memory_cache.clear()
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except OSError:
                        pass
            print("🗑️ Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache status"""
        cache_info = {
            'enabled': self.enable_cache,
            'duration_hours': self.cache_duration_hours,
            'memory_entries': len(self._memory_cache),
            'cache_dir': self.cache_dir,
            'disk_entries': 0,
            'total_size_mb': 0
        }
        
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            cache_info['disk_entries'] = len(cache_files)
            
            total_size = 0
            for filename in cache_files:
                try:
                    total_size += os.path.getsize(os.path.join(self.cache_dir, filename))
                except OSError:
                    pass
            cache_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return cache_info
    
    def screen_stocks(self, universe: str, strategy_key: str) -> Dict[str, Any]:
        """
        Screen stocks from universe using specified strategy
        
        Args:
            universe: Stock universe (nifty50, nifty500, fno)
            strategy_key: Strategy identifier
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        try:
            # Get stocks from universe
            print(f"Fetching stocks from {universe} universe...")
            stock_symbols = self.data_manager.get_universe_stocks(universe)
            
            if not stock_symbols:
                self.audit_logger.log_error("universe_fetch", universe, "No stocks found in universe")
                return {
                    'success': False,
                    'error': f'No stocks found for universe: {universe}',
                    'passing_stocks': [],
                    'total_stocks': 0
                }
            
            print(f"Found {len(stock_symbols)} stocks. Fetching data...")
            
            # Try to get cached data first
            stocks_data = self._get_cached_data(universe)
            cache_used = stocks_data is not None
            
            # Log screening start
            data_source = self.data_manager.get_current_provider_name()
            self.audit_logger.log_screening_start(
                universe, strategy_key, len(stock_symbols), data_source, cache_used
            )
            
            if stocks_data is None:
                # Cache miss - fetch fresh data
                print("🔄 Fetching fresh data...")
                self.audit_logger.log_cache_operation("miss", universe)
                stocks_data = []
                
                for i, symbol in enumerate(stock_symbols):
                    print(f"Processing {symbol} ({i+1}/{len(stock_symbols)})")
                    try:
                        stock_data = self.data_manager.get_stock_data(symbol)
                        if stock_data['quote']:  # Only add if data was fetched successfully
                            stocks_data.append(stock_data)
                            
                            # Log stock data
                            historical_count = len(stock_data.get('historical', []))
                            self.audit_logger.log_stock_data(
                                symbol, stock_data['quote'], historical_count, 
                                stock_data.get('source', data_source)
                            )
                        else:
                            self.audit_logger.log_error("data_fetch", symbol, "No quote data available")
                    except Exception as e:
                        self.audit_logger.log_error("data_fetch", symbol, str(e))
                
                # Cache the fetched data
                self._cache_data(universe, stocks_data)
            else:
                # Filter cached data to only include stocks in current universe
                # (in case universe composition changed)
                stocks_data = [stock for stock in stocks_data if stock['symbol'] in stock_symbols]
                
                # Log cached stock data
                for stock_data in stocks_data:
                    historical_count = len(stock_data.get('historical', []))
                    self.audit_logger.log_stock_data(
                        stock_data['symbol'], stock_data['quote'], historical_count, 
                        f"{stock_data.get('source', data_source)}_cached"
                    )
            
            # Handle momentum ranking separately
            if strategy_key == 'momentum_ranking':
                print("Calculating momentum scores and ranking...")
                ranked_stocks = self.strategy_manager.run_momentum_ranking_with_logging(
                    stocks_data, top_n=10, audit_logger=self.audit_logger
                )
                
                execution_time = time.time() - start_time
                self.audit_logger.log_momentum_results(
                    universe, len(stocks_data), ranked_stocks, execution_time
                )
                self.audit_logger.log_session_end(True, f"Momentum ranking completed: {len(ranked_stocks)} stocks ranked")
                
                return {
                    'success': True,
                    'ranked_stocks': ranked_stocks,
                    'total_stocks': len(stocks_data),
                    'universe': universe,
                    'strategy': strategy_key,
                    'strategy_info': self.strategy_manager.get_strategy(strategy_key).get_info()
                }
            else:
                # Apply regular strategy
                print(f"Applying strategy: {strategy_key}")
                passing_stocks = self.strategy_manager.run_strategy_with_logging(
                    strategy_key, stocks_data, audit_logger=self.audit_logger
                )
                
                execution_time = time.time() - start_time
                self.audit_logger.log_screening_results(
                    universe, strategy_key, len(stocks_data), len(passing_stocks), 
                    passing_stocks, execution_time
                )
                self.audit_logger.log_session_end(True, f"Strategy screening completed: {len(passing_stocks)} stocks passed")
                
                return {
                    'success': True,
                    'passing_stocks': passing_stocks,
                    'stocks_data': stocks_data,  # Include stock data for price display
                    'total_stocks': len(stocks_data),
                    'universe': universe,
                    'strategy': strategy_key,
                    'strategy_info': self.strategy_manager.get_strategy(strategy_key).get_info()
                }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.audit_logger.log_error("screening", universe, str(e))
            self.audit_logger.log_session_end(False, f"Screening failed after {execution_time:.2f}s: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'passing_stocks': [],
                'total_stocks': 0
            }
    
    def get_audit_log_stats(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        return self.audit_logger.get_log_stats()