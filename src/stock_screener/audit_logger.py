"""
Audit logging system for stock screener calculations and data verification
"""
import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class AuditLogger:
    """Audit logger for stock screener with automatic log rotation"""
    
    def __init__(self, log_dir: str = "logs", max_bytes: int = 2 * 1024 * 1024, backup_count: int = 10):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory to store log files
            max_bytes: Maximum size of each log file (default: 2MB)
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('stock_screener_audit')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create rotating file handler
        log_file = self.log_dir / "stock_screener_audit.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
        
        # Also log to console for debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_screening_start(self, universe: str, strategy: str, total_stocks: int, 
                           data_source: str, cache_used: bool = False):
        """Log the start of a screening session"""
        self.logger.info(f"SESSION_START | ID: {self.session_id}")
        self.logger.info(f"SCREENING_CONFIG | Universe: {universe} | Strategy: {strategy} | "
                        f"Total_Stocks: {total_stocks} | Data_Source: {data_source} | "
                        f"Cache_Used: {cache_used}")
    
    def log_stock_data(self, symbol: str, quote_data: Dict[str, Any], 
                      historical_count: int, data_source: str):
        """Log stock data retrieval"""
        price = quote_data.get('lastPrice', 0)
        volume = quote_data.get('totalTradedVolume', 0)
        change_pct = quote_data.get('pChange', 0)
        
        self.logger.info(f"STOCK_DATA | Symbol: {symbol} | Price: {price:.2f} | "
                        f"Volume: {volume} | Change%: {change_pct:.2f} | "
                        f"Historical_Days: {historical_count} | Source: {data_source}")
    
    def log_technical_calculation(self, symbol: str, indicator: str, 
                                 value: float, parameters: Dict[str, Any] = None):
        """Log technical indicator calculations"""
        params_str = json.dumps(parameters) if parameters else "{}"
        self.logger.info(f"TECHNICAL_CALC | Symbol: {symbol} | Indicator: {indicator} | "
                        f"Value: {value:.4f} | Parameters: {params_str}")
    
    def log_strategy_evaluation(self, symbol: str, strategy: str, 
                               passed: bool, criteria: Dict[str, Any] = None):
        """Log strategy evaluation for each stock"""
        criteria_str = json.dumps(criteria) if criteria else "{}"
        result = "PASS" if passed else "FAIL"
        self.logger.info(f"STRATEGY_EVAL | Symbol: {symbol} | Strategy: {strategy} | "
                        f"Result: {result} | Criteria: {criteria_str}")
    
    def log_momentum_calculation(self, symbol: str, individual_scores: Dict[str, float], 
                                total_score: float, rank: Optional[int] = None):
        """Log momentum calculation details"""
        scores_str = json.dumps({k: round(v, 4) for k, v in individual_scores.items()})
        rank_str = f" | Rank: {rank}" if rank is not None else ""
        self.logger.info(f"MOMENTUM_CALC | Symbol: {symbol} | Total_Score: {total_score:.4f} | "
                        f"Individual_Scores: {scores_str}{rank_str}")
    
    def log_composite_strategy(self, symbol: str, strategy_results: List[Dict[str, Any]], 
                              operators: List[str], final_result: bool):
        """Log composite strategy evaluation"""
        results_str = json.dumps(strategy_results)
        operators_str = " ".join(operators)
        result = "PASS" if final_result else "FAIL"
        self.logger.info(f"COMPOSITE_EVAL | Symbol: {symbol} | Strategy_Results: {results_str} | "
                        f"Operators: {operators_str} | Final_Result: {result}")
    
    def log_screening_results(self, universe: str, strategy: str, 
                             total_analyzed: int, passing_count: int, 
                             passing_stocks: List[str], execution_time: float):
        """Log final screening results"""
        stocks_str = ", ".join(passing_stocks) if len(passing_stocks) <= 20 else f"{len(passing_stocks)} stocks"
        self.logger.info(f"SCREENING_RESULTS | Universe: {universe} | Strategy: {strategy} | "
                        f"Total_Analyzed: {total_analyzed} | Passing_Count: {passing_count} | "
                        f"Execution_Time: {execution_time:.2f}s")
        self.logger.info(f"PASSING_STOCKS | {stocks_str}")
    
    def log_momentum_results(self, universe: str, total_analyzed: int, 
                            top_stocks: List[Dict[str, Any]], execution_time: float):
        """Log momentum ranking results"""
        self.logger.info(f"MOMENTUM_RESULTS | Universe: {universe} | "
                        f"Total_Analyzed: {total_analyzed} | "
                        f"Execution_Time: {execution_time:.2f}s")
        
        for i, stock in enumerate(top_stocks[:10], 1):
            self.logger.info(f"MOMENTUM_RANK_{i} | Symbol: {stock['symbol']} | "
                           f"Score: {stock['momentum_score']:.4f} | "
                           f"Price: {stock['current_price']:.2f}")
    
    def log_error(self, context: str, symbol: str, error: str):
        """Log errors during processing"""
        self.logger.error(f"ERROR | Context: {context} | Symbol: {symbol} | Error: {error}")
    
    def log_cache_operation(self, operation: str, universe: str, 
                           hit: bool = None, size: int = None):
        """Log cache operations"""
        if operation == "hit":
            self.logger.info(f"CACHE_HIT | Universe: {universe} | Size: {size} stocks")
        elif operation == "miss":
            self.logger.info(f"CACHE_MISS | Universe: {universe}")
        elif operation == "store":
            self.logger.info(f"CACHE_STORE | Universe: {universe} | Size: {size} stocks")
        elif operation == "clear":
            self.logger.info(f"CACHE_CLEAR | Universe: {universe}")
    
    def log_data_validation(self, symbol: str, validation_type: str, 
                           passed: bool, details: str = ""):
        """Log data validation results"""
        result = "PASS" if passed else "FAIL"
        self.logger.info(f"DATA_VALIDATION | Symbol: {symbol} | Type: {validation_type} | "
                        f"Result: {result} | Details: {details}")
    
    def log_session_end(self, success: bool, summary: str = ""):
        """Log the end of a screening session"""
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(f"SESSION_END | ID: {self.session_id} | Status: {status} | "
                        f"Summary: {summary}")
        self.logger.info("-" * 100)  # Separator between sessions
    
    def get_log_files(self) -> List[Path]:
        """Get list of all log files"""
        return list(self.log_dir.glob("stock_screener_audit.log*"))
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log files"""
        log_files = self.get_log_files()
        total_size = sum(f.stat().st_size for f in log_files if f.exists())
        
        return {
            'log_dir': str(self.log_dir),
            'file_count': len(log_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'current_session': self.session_id,
            'files': [{'name': f.name, 'size_kb': round(f.stat().st_size / 1024, 1)} 
                     for f in log_files if f.exists()]
        }

# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def log_screening_start(universe: str, strategy: str, total_stocks: int, 
                       data_source: str, cache_used: bool = False):
    """Convenience function for logging screening start"""
    get_audit_logger().log_screening_start(universe, strategy, total_stocks, data_source, cache_used)

def log_stock_data(symbol: str, quote_data: Dict[str, Any], 
                  historical_count: int, data_source: str):
    """Convenience function for logging stock data"""
    get_audit_logger().log_stock_data(symbol, quote_data, historical_count, data_source)

def log_technical_calculation(symbol: str, indicator: str, 
                             value: float, parameters: Dict[str, Any] = None):
    """Convenience function for logging technical calculations"""
    get_audit_logger().log_technical_calculation(symbol, indicator, value, parameters)

def log_strategy_evaluation(symbol: str, strategy: str, 
                           passed: bool, criteria: Dict[str, Any] = None):
    """Convenience function for logging strategy evaluation"""
    get_audit_logger().log_strategy_evaluation(symbol, strategy, passed, criteria)

def log_momentum_calculation(symbol: str, individual_scores: Dict[str, float], 
                            total_score: float, rank: Optional[int] = None):
    """Convenience function for logging momentum calculations"""
    get_audit_logger().log_momentum_calculation(symbol, individual_scores, total_score, rank)

def log_error(context: str, symbol: str, error: str):
    """Convenience function for logging errors"""
    get_audit_logger().log_error(context, symbol, error)

def log_cache_operation(operation: str, universe: str, 
                       hit: bool = None, size: int = None):
    """Convenience function for logging cache operations"""
    get_audit_logger().log_cache_operation(operation, universe, hit, size)