# Stock Screener App

A Python-based stock screener application with Streamlit frontend that screens NSE stocks using various technical analysis strategies.

## Features

- **Multiple Stock Universes**: NIFTY 50, NIFTY 500, F&O stocks
- **Multiple Data Sources**: YFinance (default) and NSE data with automatic fallback
- **Momentum Screener**: Comprehensive momentum analysis using multiple technical indicators
- **Environment Configuration**: Easy setup via .env file
- **Interactive UI**: Clean Streamlit interface with data provider switching

## Current Strategies

1. **Above VWAP**: Finds stocks trading above Volume Weighted Average Price
2. **Near 50-day EMA**: Finds stocks within 5% of their 50-day Exponential Moving Average
3. **Momentum Ranking**: Ranks stocks by comprehensive momentum score using:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - ROC (Rate of Change)
   - Stochastic Oscillator
   - Williams %R
   - Moving Average Position
4. **High Volume**: Stocks with volume above average (configurable multiplier)
5. **Breakout**: Stocks breaking above recent highs (configurable lookback period)
6. **RSI Oversold/Overbought**: RSI-based filtering with configurable thresholds
7. **Strategy Chaining**: Combine multiple strategies with AND/OR logic:
   - Strong Momentum: RSI > 60 AND High Volume
   - Trend + Breakout: Above 50 EMA AND Breakout
   - Quality Setup: (Above VWAP OR High Volume) AND RSI not overbought

## Installation & Usage

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Quick Start

1. **Setup configuration:**
   ```bash
   cp .env.example .env
   # Edit .env file to configure data providers
   ```

2. **Install dependencies and run the app:**
   ```bash
   cd stock-screener
   uv run streamlit run src/stock_screener/app.py
   ```

3. **Test momentum screener:**
   ```bash
   uv run python momentum_screener_test.py
   ```

4. **Install in development mode:**
   ```bash
   cd stock-screener
   uv sync
   ```

### Development

- **Add new dependencies:**
  ```bash
  uv add package-name
  ```

- **Add development dependencies:**
  ```bash
  uv add --dev pytest black flake8
  ```

- **Run the CLI version:**
  ```bash
  uv run stock-screener
  ```

## Architecture

- `src/stock_screener/data_sources.py`: Consolidated data providers (YFinance, NSE) and DataManager with caching
- `src/stock_screener/strategy.py`: All screening strategies including momentum analysis and chaining
- `src/stock_screener/screener.py`: Main orchestrator with caching and audit logging
- `src/stock_screener/audit_logger.py`: Comprehensive audit logging system
- `src/stock_screener/app.py`: Streamlit frontend with provider switching and log viewer
- `.env`: Environment configuration for data providers, caching, and logging

## Strategy Framework

The framework makes it easy to add new strategies with interactive parameters and strategy chaining capabilities.

### Quick Start

1. **Copy the template**: Use `strategy_template.py` as your starting point
2. **Implement your logic**: Modify the `apply()` method with your strategy
3. **Configure parameters**: Define UI controls in `get_parameter_config()`
4. **Register strategy**: Add it to `StrategyManager._register_default_strategies()`
5. **Done!** Your strategy appears automatically in the Streamlit UI

### Strategy Types

#### Filter Strategies (`StrategyType.FILTER`)
- **Purpose**: Pass/fail filtering of stocks
- **Returns**: `bool` (True if stock passes criteria)
- **UI Result**: List of stocks that passed
- **Example**: "Stocks above VWAP", "RSI Oversold"
- **Chainable**: ✅ Yes

#### Ranking Strategies (`StrategyType.RANKING`)
- **Purpose**: Rank stocks by score
- **Returns**: Custom ranking logic (handled separately)
- **UI Result**: Ranked list with scores
- **Example**: "Momentum Ranking"
- **Chainable**: ❌ No

#### Composite Strategies (`StrategyType.COMPOSITE`)
- **Purpose**: Combine multiple filter strategies with logical operators
- **Returns**: `bool` (result of combined logic)
- **UI Result**: List of stocks that passed all conditions
- **Example**: "RSI > 60 AND High Volume"
- **Chainable**: ✅ Yes

### Strategy Categories

Organize your strategies by category for better UI:

- `StrategyCategory.MOMENTUM` - RSI, MACD, momentum-based
- `StrategyCategory.TREND` - Moving averages, trend following
- `StrategyCategory.VOLUME` - Volume-based strategies
- `StrategyCategory.VOLATILITY` - Volatility-based strategies
- `StrategyCategory.VALUE` - Fundamental value strategies
- `StrategyCategory.TECHNICAL` - General technical analysis

### Parameter Configuration

Make your strategies interactive with parameter controls:

```python
def get_parameter_config(self) -> Dict[str, Dict[str, Any]]:
    return {
        "my_threshold": {
            "type": "slider",           # Control type
            "min": 0,                   # Minimum value
            "max": 100,                 # Maximum value
            "default": 50,              # Default value
            "step": 5,                  # Step size
            "label": "My Threshold",    # Display label
            "help": "Explanation text"  # Help tooltip
        }
    }
```

#### Available Control Types

**Slider:**
```python
{
    "type": "slider",
    "min": 0.0, "max": 100.0, "default": 50.0, "step": 0.1,
    "label": "Threshold Value", "help": "Adjust the threshold"
}
```

**Select Box:**
```python
{
    "type": "selectbox",
    "options": ["option1", "option2", "option3"], "default": "option1",
    "label": "Choose Option", "help": "Select from available options"
}
```

**Number Input:**
```python
{
    "type": "number_input",
    "min": 1, "max": 1000, "default": 10, "step": 1,
    "label": "Number Value", "help": "Enter a number"
}
```

### Data Structure

Your strategy receives stock data in this format:

```python
stock_data = {
    'symbol': 'RELIANCE',
    'quote': {
        'lastPrice': 2500.0, 'averagePrice': 2480.0, 'change': 20.0,
        'pChange': 0.8, 'totalTradedVolume': 1000000, 'marketCap': 1500000000000
    },
    'historical': [
        {
            'CH_TIMESTAMP': '22-Oct-2024', 'CH_CLOSING_PRICE': 2500.0,
            'CH_TRADE_HIGH_PRICE': 2520.0, 'CH_TRADE_LOW_PRICE': 2480.0,
            'CH_OPENING_PRICE': 2490.0, 'CH_TOT_TRADED_QTY': 500000
        }
        # ... more historical data
    ],
    'source': 'yfinance'
}
```

### Example Strategy

```python
class MyBollingerStrategy(Strategy):
    def __init__(self, std_dev: float = 2.0, period: int = 20):
        super().__init__(
            name=f"Bollinger Breakout ({std_dev}σ, {period}d)",
            description="Stocks breaking above upper Bollinger Band",
            strategy_type=StrategyType.FILTER,
            category=StrategyCategory.VOLATILITY,
            parameters={"std_dev": std_dev, "period": period},
            min_data_points=period + 5,
            tags=["bollinger", "breakout", "volatility"]
        )
        self.std_dev = std_dev
        self.period = period
    
    def get_parameter_config(self):
        return {
            "std_dev": {
                "type": "slider", "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1,
                "label": "Standard Deviations", "help": "Number of standard deviations for bands"
            },
            "period": {
                "type": "slider", "min": 10, "max": 50, "default": 20, "step": 5,
                "label": "Period", "help": "Moving average period"
            }
        }
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        try:
            historical = stock_data.get('historical')
            quote = stock_data.get('quote')
            
            if not historical or not quote:
                return False
            
            df = pd.DataFrame(historical)
            df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
            df = df.dropna()
            
            if len(df) < self.period:
                return False
            
            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df['close'], window=self.period, window_dev=self.std_dev)
            upper_band = bb.bollinger_hband()
            current_price = float(quote.get('lastPrice', 0))
            
            return current_price > upper_band.iloc[-1]
            
        except (ValueError, TypeError, KeyError):
            return False

# Register in StrategyManager
def _register_default_strategies(self):
    # ... existing strategies ...
    self.register_strategy('bollinger_breakout', MyBollingerStrategy())
```

### Strategy Chaining

Combine multiple filter strategies with logical operators (AND, OR):

```python
# Using Chain Builder
builder = strategy_manager.create_chain_builder()
composite = (builder
    .add_strategy('rsi_oversold', rsi_lower=60, condition="neutral")
    .add_strategy('high_volume', LogicalOperator.AND, volume_multiplier=2.0)
    .add_strategy('above_vwap', LogicalOperator.AND)
    .build("Strong Momentum Setup"))

# Using Configuration
configs = [
    {'key': 'near_50_ema', 'parameters': {'tolerance_percent': 2.0}},
    {'key': 'rsi_oversold', 'operator': 'AND', 'parameters': {'rsi_lower': 50}},
    {'key': 'high_volume', 'operator': 'OR', 'parameters': {'volume_multiplier': 1.5}}
]
composite = strategy_manager.create_composite_strategy("My Chain", configs)
```

#### Streamlit UI Chaining

1. Select "Chain Strategies" mode in the sidebar
2. Add strategies one by one with parameters
3. Choose AND/OR operators between strategies
4. View your chain in real-time
5. Run the composite strategy

### Best Practices

1. **Error Handling**: Always wrap logic in try-catch blocks
2. **Data Validation**: Check data availability and quality
3. **Performance**: Use vectorized operations, set appropriate `min_data_points`
4. **Documentation**: Use clear names, descriptions, and helpful parameter descriptions

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify:

- `DEFAULT_DATA_PROVIDER`: Primary data source (yfinance/nse)
- `FALLBACK_DATA_PROVIDER`: Backup data source
- `REQUEST_TIMEOUT`: API request timeout
- `RETRY_ATTEMPTS`: Number of retry attempts

## Data Providers

- **YFinance**: Default provider, reliable for Indian stocks with .NS suffix
- **NSE**: Fallback provider using nsepython library
- **Automatic Fallback**: Switches to backup provider if primary fails

## Performance & Caching

The application includes intelligent caching to improve performance:

- **Multi-Level Caching**: Memory and disk-based caching with automatic fallback
- **Smart Cache Management**: Time-based expiration (4 hours default)
- **Provider-Aware**: Different cache for different data providers
- **Performance Boost**: 10-30x faster when switching between strategies on same universe
- **Cache Controls**: View, clear, and manage cache through Streamlit interface

### Cache Configuration
```bash
# In .env file
ENABLE_CACHE=true
CACHE_DURATION_HOURS=4
```

## Audit Logging & Verification

Comprehensive audit logging system for data verification and compliance:

- **Complete Traceability**: Every calculation and data point logged
- **Automatic Log Rotation**: 2MB max file size with backup management
- **Structured Logging**: Session-based organization with unique IDs
- **Verification Support**: Trace why each stock passed/failed criteria
- **Error Tracking**: Detailed error logging for debugging

### Log Categories
- **Session Management**: Start/end of screening runs
- **Data Verification**: Stock prices, volumes, historical data validation
- **Technical Calculations**: All indicator calculations (RSI, MACD, etc.)
- **Strategy Evaluation**: Pass/fail decisions with criteria
- **Results & Rankings**: Final screening results and momentum rankings
- **Cache Operations**: Performance tracking and cache usage

### Log Viewer
- View logs directly in Streamlit interface
- Download log files for external analysis
- Real-time log statistics and session tracking
- Adjustable log content display (last N lines)

## Testing

The project includes comprehensive test scripts:

```bash
# Test basic functionality
python simple_app_test.py

# Test table format display
python test_table_format.py

# Test caching performance
python test_caching.py

# Test audit logging
python test_audit_logging.py

# Test strategy methods
python test_strategy_methods.py

# Test strategy chaining
python strategy_chaining_example.py
```