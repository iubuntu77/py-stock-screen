"""
Strategy module for stock screening strategies
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import ta
from enum import Enum

class StrategyType(Enum):
    """Types of strategies for UI categorization"""
    FILTER = "filter"           # Returns pass/fail for each stock
    RANKING = "ranking"         # Returns ranked list of stocks
    SCORING = "scoring"         # Returns scores for all stocks
    COMPOSITE = "composite"     # Combines multiple strategies

class StrategyCategory(Enum):
    """Categories for organizing strategies in UI"""
    MOMENTUM = "Momentum"
    TREND = "Trend"
    VOLUME = "Volume"
    VOLATILITY = "Volatility"
    VALUE = "Value"
    TECHNICAL = "Technical"
    COMPOSITE = "Composite"
    CUSTOM = "Custom"

class LogicalOperator(Enum):
    """Logical operators for combining strategies"""
    AND = "AND"
    OR = "OR"

class Strategy(ABC):
    """Abstract base class for screening strategies"""
    
    def __init__(
        self, 
        name: str, 
        description: str,
        strategy_type: StrategyType = StrategyType.FILTER,
        category: StrategyCategory = StrategyCategory.TECHNICAL,
        parameters: Optional[Dict[str, Any]] = None,
        min_data_points: int = 20,
        tags: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.strategy_type = strategy_type
        self.category = category
        self.parameters = parameters or {}
        self.min_data_points = min_data_points
        self.tags = tags or []
    
    @abstractmethod
    def apply(self, stock_data: Dict[str, Any]) -> Union[bool, float, Dict[str, Any]]:
        """Apply strategy to stock data and return result based on strategy type"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.strategy_type.value,
            'category': self.category.value,
            'parameters': self.parameters,
            'min_data_points': self.min_data_points,
            'tags': self.tags
        }
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter configuration for UI generation"""
        return {}
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
    
    def validate_data(self, stock_data: Dict[str, Any]) -> bool:
        """Validate if stock data is sufficient for this strategy"""
        historical = stock_data.get('historical', [])
        if isinstance(historical, list):
            return len(historical) >= self.min_data_points
        elif isinstance(historical, pd.DataFrame):
            return len(historical) >= self.min_data_points
        return False

class AboveVWAPStrategy(Strategy):
    """Strategy to find stocks trading above VWAP"""
    
    def __init__(self):
        super().__init__(
            name="Above VWAP",
            description="Stocks trading above Volume Weighted Average Price",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.VOLUME,
            min_data_points=1,
            tags=["vwap", "volume", "price_action"]
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
            description=f"Stocks within {tolerance_percent}% of 50-day Exponential Moving Average",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.TREND,
            parameters={"tolerance_percent": tolerance_percent},
            min_data_points=50,
            tags=["ema", "moving_average", "trend"]
        )
        self.tolerance_percent = tolerance_percent
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter configuration for UI"""
        return {
            "tolerance_percent": {
                "type": "slider",
                "min": 1.0,
                "max": 20.0,
                "default": 5.0,
                "step": 0.5,
                "label": "Tolerance Percentage (%)",
                "help": "Maximum percentage deviation from 50-day EMA"
            }
        }    

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

class MomentumStrategy(Strategy):
    """Strategy to calculate momentum score and rank stocks"""
    
    def __init__(self):
        super().__init__(
            name="Momentum Ranking",
            description="Ranks stocks based on multiple momentum indicators (RSI, MACD, Price Rate of Change)",
            strategy_type=StrategyType.RANKING,
            category=StrategyCategory.MOMENTUM,
            min_data_points=30,
            tags=["momentum", "rsi", "macd", "roc", "ranking"]
        )
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """For momentum strategy, we'll use this to calculate score instead of filtering"""
        return True  # We want all stocks to calculate momentum scores
    
    def calculate_momentum_score(self, stock_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive momentum score for a stock
        Returns: (total_score, individual_scores_dict)
        """
        try:
            historical = stock_data.get('historical')
            if not historical:
                return 0.0, {}
            
            # Convert to DataFrame
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            if len(df) < 30:  # Need at least 30 days for calculations
                return 0.0, {}
            
            # Prepare price data
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            df['high'] = pd.to_numeric(df['CH_TRADE_HIGH_PRICE'], errors='coerce')
            df['low'] = pd.to_numeric(df['CH_TRADE_LOW_PRICE'], errors='coerce')
            df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            if len(df) < 20:
                return 0.0, {}
            
            scores = {}
            
            # 1. RSI Score (0-100, normalize to 0-1)
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            # RSI momentum: higher RSI (but not overbought) = better momentum
            if current_rsi > 70:
                rsi_score = 0.7  # Overbought penalty
            elif current_rsi > 50:
                rsi_score = (current_rsi - 50) / 20  # Scale 50-70 to 0-1
            else:
                rsi_score = 0  # Below 50 = no momentum
            scores['rsi'] = rsi_score
            
            # 2. MACD Score
            macd_indicator = ta.trend.MACD(close=df['close'])
            macd_line = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            if not pd.isna(macd_histogram.iloc[-1]):
                # Positive MACD histogram = bullish momentum
                macd_score = max(0, min(1, (macd_histogram.iloc[-1] + abs(macd_histogram.iloc[-1])) / (2 * abs(macd_histogram.iloc[-1]) + 0.001)))
            else:
                macd_score = 0
            scores['macd'] = macd_score
            
            # 3. Price Rate of Change (ROC) - 10 day
            roc = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()
            current_roc = roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0
            # Normalize ROC: positive = good, scale to 0-1
            roc_score = max(0, min(1, (current_roc + 10) / 20))  # Assume -10% to +10% range
            scores['roc'] = roc_score
            
            # 4. Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            stoch_k = stoch.stoch()
            current_stoch = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
            # Similar to RSI logic
            if current_stoch > 80:
                stoch_score = 0.7  # Overbought penalty
            elif current_stoch > 50:
                stoch_score = (current_stoch - 50) / 30  # Scale 50-80 to 0-1
            else:
                stoch_score = 0
            scores['stochastic'] = stoch_score
            
            # 5. Williams %R
            williams_r = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            current_williams = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
            # Williams %R: -20 to 0 is overbought, -100 to -80 is oversold
            # We want momentum, so -50 to -20 is good
            if current_williams > -20:
                williams_score = 0.7  # Overbought
            elif current_williams > -50:
                williams_score = (current_williams + 50) / 30  # Scale -50 to -20 to 0-1
            else:
                williams_score = 0
            scores['williams_r'] = williams_score
            
            # 6. Price vs Moving Averages
            sma_20 = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(close=df['close'], window=min(50, len(df))).sma_indicator()
            
            current_price = df['close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            sma_50_current = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            
            # Price above both MAs = strong momentum
            ma_score = 0
            if current_price > sma_20_current:
                ma_score += 0.5
            if current_price > sma_50_current:
                ma_score += 0.5
            scores['moving_average'] = ma_score
            
            # Calculate weighted total score
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'roc': 0.2,
                'stochastic': 0.15,
                'williams_r': 0.1,
                'moving_average': 0.1
            }
            
            total_score = sum(scores[key] * weights[key] for key in scores.keys())
            
            return total_score, scores
            
        except Exception as e:
            print(f"Error calculating momentum for {stock_data.get('symbol', 'Unknown')}: {e}")
            return 0.0, {}
    
    def calculate_momentum_score_with_logging(self, stock_data: Dict[str, Any], audit_logger=None) -> Tuple[float, Dict[str, float]]:
        """Calculate momentum score with detailed logging"""
        symbol = stock_data.get('symbol', 'Unknown')
        
        try:
            historical = stock_data.get('historical')
            if not historical:
                if audit_logger:
                    audit_logger.log_data_validation(symbol, "historical_data", False, "No historical data")
                return 0.0, {}
            
            # Convert to DataFrame
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            if len(df) < 30:
                if audit_logger:
                    audit_logger.log_data_validation(symbol, "data_points", False, f"Only {len(df)} days available")
                return 0.0, {}
            
            # Prepare price data
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            df['high'] = pd.to_numeric(df['CH_TRADE_HIGH_PRICE'], errors='coerce')
            df['low'] = pd.to_numeric(df['CH_TRADE_LOW_PRICE'], errors='coerce')
            df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            if len(df) < 20:
                if audit_logger:
                    audit_logger.log_data_validation(symbol, "clean_data", False, f"Only {len(df)} clean data points")
                return 0.0, {}
            
            if audit_logger:
                audit_logger.log_data_validation(symbol, "data_quality", True, f"{len(df)} clean data points")
            
            scores = {}
            
            # 1. RSI Score
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            if current_rsi > 70:
                rsi_score = 0.7
            elif current_rsi > 50:
                rsi_score = (current_rsi - 50) / 20
            else:
                rsi_score = 0
            scores['rsi'] = rsi_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "RSI", current_rsi, {"window": 14})
            
            # 2. MACD Score
            macd_indicator = ta.trend.MACD(close=df['close'])
            macd_histogram = macd_indicator.macd_diff()
            
            if not pd.isna(macd_histogram.iloc[-1]):
                macd_score = max(0, min(1, (macd_histogram.iloc[-1] + abs(macd_histogram.iloc[-1])) / (2 * abs(macd_histogram.iloc[-1]) + 0.001)))
            else:
                macd_score = 0
            scores['macd'] = macd_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "MACD_Histogram", macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0)
            
            # 3. ROC Score
            roc = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()
            current_roc = roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0
            roc_score = max(0, min(1, (current_roc + 10) / 20))
            scores['roc'] = roc_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "ROC", current_roc, {"window": 10})
            
            # 4. Stochastic Score
            stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            stoch_k = stoch.stoch()
            current_stoch = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
            
            if current_stoch > 80:
                stoch_score = 0.7
            elif current_stoch > 50:
                stoch_score = (current_stoch - 50) / 30
            else:
                stoch_score = 0
            scores['stochastic'] = stoch_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "Stochastic", current_stoch)
            
            # 5. Williams %R
            williams_r = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            current_williams = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
            
            if current_williams > -20:
                williams_score = 0.7
            elif current_williams > -50:
                williams_score = (current_williams + 50) / 30
            else:
                williams_score = 0
            scores['williams_r'] = williams_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "Williams_R", current_williams)
            
            # 6. Moving Average Score
            sma_20 = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(close=df['close'], window=min(50, len(df))).sma_indicator()
            
            current_price = df['close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            sma_50_current = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            
            ma_score = 0
            if current_price > sma_20_current:
                ma_score += 0.5
            if current_price > sma_50_current:
                ma_score += 0.5
            scores['moving_average'] = ma_score
            
            if audit_logger:
                audit_logger.log_technical_calculation(symbol, "SMA_20", sma_20_current, {"window": 20})
                audit_logger.log_technical_calculation(symbol, "SMA_50", sma_50_current, {"window": 50})
            
            # Calculate weighted total score
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'roc': 0.2,
                'stochastic': 0.15,
                'williams_r': 0.1,
                'moving_average': 0.1
            }
            
            total_score = sum(scores[key] * weights[key] for key in scores.keys())
            
            return total_score, scores
            
        except Exception as e:
            if audit_logger:
                audit_logger.log_error("momentum_calculation", symbol, str(e))
            return 0.0, {}

class HighVolumeStrategy(Strategy):
    """Strategy to find stocks with high trading volume"""
    
    def __init__(self, volume_multiplier: float = 2.0):
        super().__init__(
            name=f"High Volume ({volume_multiplier}x)",
            description=f"Stocks with volume {volume_multiplier}x above 20-day average",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.VOLUME,
            parameters={"volume_multiplier": volume_multiplier},
            min_data_points=20,
            tags=["volume", "activity", "liquidity"]
        )
        self.volume_multiplier = volume_multiplier
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "volume_multiplier": {
                "type": "slider",
                "min": 1.0,
                "max": 5.0,
                "default": 2.0,
                "step": 0.1,
                "label": "Volume Multiplier",
                "help": "How many times above average volume"
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
            
            if len(df) < 20:
                return False
            
            df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
            avg_volume = df['volume'].tail(20).mean()
            current_volume = float(quote.get('totalTradedVolume', 0))
            
            return current_volume > (avg_volume * self.volume_multiplier)
            
        except (ValueError, TypeError, KeyError):
            return False

class BreakoutStrategy(Strategy):
    """Strategy to find stocks breaking out of 20-day high"""
    
    def __init__(self, lookback_days: int = 20):
        super().__init__(
            name=f"Breakout ({lookback_days} days)",
            description=f"Stocks breaking above {lookback_days}-day high",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.MOMENTUM,
            parameters={"lookback_days": lookback_days},
            min_data_points=lookback_days + 5,
            tags=["breakout", "momentum", "price_action"]
        )
        self.lookback_days = lookback_days
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "lookback_days": {
                "type": "slider",
                "min": 10,
                "max": 100,
                "default": 20,
                "step": 5,
                "label": "Lookback Days",
                "help": "Number of days to look back for high"
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
            
            if len(df) < self.lookback_days + 1:
                return False
            
            df['high'] = pd.to_numeric(df['CH_TRADE_HIGH_PRICE'], errors='coerce')
            
            # Get the highest high in the lookback period (excluding today)
            lookback_high = df['high'].iloc[-(self.lookback_days+1):-1].max()
            current_price = float(quote.get('lastPrice', 0))
            
            return current_price > lookback_high
            
        except (ValueError, TypeError, KeyError):
            return False

class RSIStrategy(Strategy):
    """Strategy based on RSI levels"""
    
    def __init__(self, rsi_lower: float = 30, rsi_upper: float = 70, condition: str = "oversold"):
        conditions = {"oversold": "RSI below lower threshold", "overbought": "RSI above upper threshold", "neutral": "RSI between thresholds"}
        
        super().__init__(
            name=f"RSI {condition.title()} ({rsi_lower}-{rsi_upper})",
            description=f"Stocks with {conditions.get(condition, 'RSI condition')}",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.MOMENTUM,
            parameters={"rsi_lower": rsi_lower, "rsi_upper": rsi_upper, "condition": condition},
            min_data_points=20,
            tags=["rsi", "momentum", "oscillator"]
        )
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.condition = condition
    
    def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "rsi_lower": {
                "type": "slider",
                "min": 10,
                "max": 50,
                "default": 30,
                "step": 5,
                "label": "RSI Lower Threshold",
                "help": "Lower RSI threshold for oversold condition"
            },
            "rsi_upper": {
                "type": "slider",
                "min": 50,
                "max": 90,
                "default": 70,
                "step": 5,
                "label": "RSI Upper Threshold",
                "help": "Upper RSI threshold for overbought condition"
            },
            "condition": {
                "type": "selectbox",
                "options": ["oversold", "overbought", "neutral"],
                "default": "oversold",
                "label": "RSI Condition",
                "help": "Which RSI condition to filter for"
            }
        }
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        try:
            historical = stock_data.get('historical')
            
            if not historical:
                return False
            
            if isinstance(historical, list):
                df = pd.DataFrame(historical)
            else:
                df = historical
            
            if len(df) < 20:
                return False
            
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                return False
            
            if self.condition == "oversold":
                return current_rsi < self.rsi_lower
            elif self.condition == "overbought":
                return current_rsi > self.rsi_upper
            elif self.condition == "neutral":
                return self.rsi_lower <= current_rsi <= self.rsi_upper
            
            return False
            
        except (ValueError, TypeError, KeyError):
            return False

class CompositeStrategy(Strategy):
    """Strategy that combines multiple strategies with logical operators"""
    
    def __init__(self, name: str, strategies: List[Tuple[str, Strategy]], operators: List[LogicalOperator]):
        """
        Initialize composite strategy
        
        Args:
            name: Display name for the composite strategy
            strategies: List of (name, strategy) tuples
            operators: List of logical operators (AND/OR) between strategies
        """
        if len(strategies) < 2:
            raise ValueError("Composite strategy needs at least 2 strategies")
        if len(operators) != len(strategies) - 1:
            raise ValueError("Number of operators must be one less than number of strategies")
        
        # Create description
        strategy_names = [name for name, _ in strategies]
        op_symbols = [op.value for op in operators]
        
        description_parts = []
        for i, strategy_name in enumerate(strategy_names):
            description_parts.append(strategy_name)
            if i < len(op_symbols):
                description_parts.append(op_symbols[i])
        
        description = " ".join(description_parts)
        
        # Calculate minimum data points needed
        min_data_points = max(strategy.min_data_points for _, strategy in strategies)
        
        # Collect all tags
        all_tags = []
        for _, strategy in strategies:
            all_tags.extend(strategy.tags)
        all_tags = list(set(all_tags))  # Remove duplicates
        all_tags.append("composite")
        
        super().__init__(
            name=name,
            description=description,
            strategy_type=StrategyType.COMPOSITE,
            category=StrategyCategory.COMPOSITE,
            min_data_points=min_data_points,
            tags=all_tags
        )
        
        self.strategies = strategies
        self.operators = operators
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        """Apply composite strategy logic"""
        try:
            # Get results from all strategies
            results = []
            for strategy_name, strategy in self.strategies:
                result = strategy.apply(stock_data)
                results.append(result)
            
            # Apply logical operators
            final_result = results[0]
            
            for i, operator in enumerate(self.operators):
                next_result = results[i + 1]
                
                if operator == LogicalOperator.AND:
                    final_result = final_result and next_result
                elif operator == LogicalOperator.OR:
                    final_result = final_result or next_result
            
            return final_result
            
        except Exception as e:
            print(f"Error in CompositeStrategy: {e}")
            return False
    
    def get_strategy_details(self) -> List[Dict[str, Any]]:
        """Get details of component strategies"""
        details = []
        for name, strategy in self.strategies:
            details.append({
                'name': name,
                'strategy': strategy.get_info(),
                'type': strategy.strategy_type.value
            })
        return details

class StrategyChainBuilder:
    """Builder class for creating composite strategies easily"""
    
    def __init__(self, strategy_manager):
        self.strategy_manager = strategy_manager
        self.chain = []
        self.operators = []
    
    def add_strategy(self, strategy_key: str, operator: LogicalOperator = None, **parameters):
        """Add a strategy to the chain"""
        if strategy_key not in self.strategy_manager.strategies:
            raise ValueError(f"Strategy '{strategy_key}' not found")
        
        # For the first strategy, operator should be None
        if len(self.chain) == 0 and operator is not None:
            raise ValueError("First strategy in chain cannot have an operator")
        
        # For subsequent strategies, operator is required
        if len(self.chain) > 0 and operator is None:
            raise ValueError("Operator required for strategies after the first one")
        
        # Create strategy instance with parameters
        if parameters:
            strategy = self.strategy_manager.create_strategy_instance(strategy_key, **parameters)
        else:
            strategy = self.strategy_manager.strategies[strategy_key]
        
        if strategy is None:
            raise ValueError(f"Could not create strategy instance for '{strategy_key}'")
        
        strategy_name = strategy.name
        if parameters:
            # Add parameter info to name
            param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
            strategy_name += f" ({param_str})"
        
        self.chain.append((strategy_name, strategy))
        
        # Add operator for strategies after the first
        if operator is not None:
            self.operators.append(operator)
        
        return self
    
    def build(self, name: str) -> CompositeStrategy:
        """Build the composite strategy"""
        if len(self.chain) < 2:
            raise ValueError("Need at least 2 strategies to build composite")
        
        return CompositeStrategy(name, self.chain, self.operators)
    
    def reset(self):
        """Reset the builder"""
        self.chain = []
        self.operators = []
        return self

class StrategyManager:
    """Manager class for handling multiple strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies"""
        # Original strategies
        self.register_strategy('above_vwap', AboveVWAPStrategy())
        self.register_strategy('near_50_ema', Near50EMAStrategy())
        self.register_strategy('momentum_ranking', MomentumStrategy())
        
        # New example strategies
        self.register_strategy('high_volume', HighVolumeStrategy())
        self.register_strategy('breakout', BreakoutStrategy())
        self.register_strategy('rsi_oversold', RSIStrategy(condition="oversold"))
        self.register_strategy('rsi_overbought', RSIStrategy(condition="overbought"))
        
        # Predefined composite strategies
        try:
            self._register_composite_strategies()
        except Exception as e:
            print(f"Warning: Could not register composite strategies: {e}")
            # Continue without composite strategies for now
    
    def _register_composite_strategies(self):
        """Register predefined composite strategies"""
        try:
            # Skip composite strategies for now to avoid initialization issues
            # Will be re-enabled once the basic framework is working
            pass
            
        except Exception as e:
            print(f"Error registering composite strategies: {e}")
    
    def register_strategy(self, key: str, strategy: Strategy):
        """Register a new strategy"""
        self.strategies[key] = strategy
    
    def get_strategy(self, key: str) -> Strategy:
        """Get strategy by key"""
        return self.strategies.get(key)
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all available strategies with their info"""
        return {key: strategy.get_info() for key, strategy in self.strategies.items()}
    
    def get_strategies_by_category(self) -> Dict[str, List[Tuple[str, Strategy]]]:
        """Get strategies organized by category"""
        categories = {}
        for key, strategy in self.strategies.items():
            category = strategy.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append((key, strategy))
        return categories
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> Dict[str, Strategy]:
        """Get strategies filtered by type"""
        return {
            key: strategy for key, strategy in self.strategies.items() 
            if strategy.strategy_type == strategy_type
        }
    
    def get_filter_strategies(self) -> Dict[str, Strategy]:
        """Get only filter-type strategies"""
        return self.get_strategies_by_type(StrategyType.FILTER)
    
    def get_ranking_strategies(self) -> Dict[str, Strategy]:
        """Get only ranking-type strategies"""
        return self.get_strategies_by_type(StrategyType.RANKING)
    
    def create_strategy_instance(self, strategy_key: str, **parameters) -> Optional[Strategy]:
        """Create a new instance of a strategy with custom parameters"""
        if strategy_key not in self.strategies:
            return None
        
        strategy_class = type(self.strategies[strategy_key])
        
        # Try to create instance with parameters
        try:
            return strategy_class(**parameters)
        except TypeError:
            # Fallback to default instance
            return strategy_class()
    
    def create_chain_builder(self) -> StrategyChainBuilder:
        """Create a new strategy chain builder"""
        return StrategyChainBuilder(self)
    
    def create_composite_strategy(self, name: str, strategy_configs: List[Dict[str, Any]]) -> CompositeStrategy:
        """
        Create a composite strategy from configuration
        
        Args:
            name: Name for the composite strategy
            strategy_configs: List of strategy configurations
                Each config should have: {'key': str, 'operator': str, 'parameters': dict}
        """
        builder = self.create_chain_builder()
        
        for i, config in enumerate(strategy_configs):
            strategy_key = config['key']
            parameters = config.get('parameters', {})
            
            if i == 0:
                # First strategy doesn't need an operator
                builder.add_strategy(strategy_key, **parameters)
            else:
                operator_str = config.get('operator', 'AND')
                operator = LogicalOperator.AND if operator_str == 'AND' else LogicalOperator.OR
                builder.add_strategy(strategy_key, operator, **parameters)
        
        return builder.build(name)
    
    def get_composite_strategies(self) -> Dict[str, CompositeStrategy]:
        """Get only composite strategies"""
        return {
            key: strategy for key, strategy in self.strategies.items() 
            if isinstance(strategy, CompositeStrategy)
        }
    
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
    
    def run_momentum_ranking(self, stocks_data: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Run momentum strategy and return top N stocks ranked by momentum score
        
        Args:
            stocks_data: List of stock data dictionaries
            top_n: Number of top stocks to return
            
        Returns:
            List of dictionaries with stock symbol, score, and individual indicator scores
        """
        momentum_strategy = self.get_strategy('momentum_ranking')
        if not momentum_strategy or not isinstance(momentum_strategy, MomentumStrategy):
            raise ValueError("Momentum strategy not found or invalid")
        
        stock_scores = []
        
        for stock_data in stocks_data:
            symbol = stock_data.get('symbol', 'Unknown')
            total_score, individual_scores = momentum_strategy.calculate_momentum_score(stock_data)
            
            stock_scores.append({
                'symbol': symbol,
                'momentum_score': total_score,
                'individual_scores': individual_scores,
                'current_price': stock_data.get('quote', {}).get('lastPrice', 0)
            })
        
        # Sort by momentum score in descending order
        stock_scores.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        # Return top N stocks
        return stock_scores[:top_n]
    
    def run_strategy_with_logging(self, strategy_key: str, stocks_data: List[Dict[str, Any]], 
                                 audit_logger=None) -> List[str]:
        """Run strategy with detailed logging"""
        strategy = self.get_strategy(strategy_key)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_key}' not found")
        
        passing_stocks = []
        for stock_data in stocks_data:
            symbol = stock_data['symbol']
            try:
                passed = strategy.apply(stock_data)
                
                # Log strategy evaluation
                if audit_logger:
                    criteria = {}
                    if hasattr(strategy, 'get_evaluation_criteria'):
                        criteria = strategy.get_evaluation_criteria(stock_data)
                    audit_logger.log_strategy_evaluation(symbol, strategy_key, passed, criteria)
                
                if passed:
                    passing_stocks.append(symbol)
                    
            except Exception as e:
                if audit_logger:
                    audit_logger.log_error("strategy_evaluation", symbol, str(e))
        
        return passing_stocks
    
    def run_momentum_ranking_with_logging(self, stocks_data: List[Dict[str, Any]], 
                                         top_n: int = 10, audit_logger=None) -> List[Dict[str, Any]]:
        """Run momentum strategy with detailed logging"""
        momentum_strategy = self.get_strategy('momentum_ranking')
        if not momentum_strategy or not isinstance(momentum_strategy, MomentumStrategy):
            raise ValueError("Momentum strategy not found or invalid")
        
        stock_scores = []
        
        for stock_data in stocks_data:
            symbol = stock_data.get('symbol', 'Unknown')
            try:
                total_score, individual_scores = momentum_strategy.calculate_momentum_score_with_logging(
                    stock_data, audit_logger
                )
                
                stock_scores.append({
                    'symbol': symbol,
                    'momentum_score': total_score,
                    'individual_scores': individual_scores,
                    'current_price': stock_data.get('quote', {}).get('lastPrice', 0)
                })
                
            except Exception as e:
                if audit_logger:
                    audit_logger.log_error("momentum_calculation", symbol, str(e))
        
        # Sort by momentum score in descending order
        stock_scores.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        # Log final rankings
        for i, stock in enumerate(stock_scores[:top_n], 1):
            if audit_logger:
                audit_logger.log_momentum_calculation(
                    stock['symbol'], stock['individual_scores'], 
                    stock['momentum_score'], i
                )
        
        return stock_scores[:top_n]