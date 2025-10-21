# Stock Screener App

A Python-based stock screener application with Streamlit frontend that screens NSE stocks using various technical analysis strategies.

## Features

- **Multiple Stock Universes**: NIFTY 50, NIFTY 500, F&O stocks
- **Pluggable Strategies**: Easy to add new screening strategies
- **Real-time Data**: Fetches live data from NSE using nsepython
- **Interactive UI**: Clean Streamlit interface

## Current Strategies

1. **Above VWAP**: Finds stocks trading above Volume Weighted Average Price
2. **Near 50-day EMA**: Finds stocks within 5% of their 50-day Exponential Moving Average

## Installation & Usage

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Quick Start

1. **Install dependencies and run the app:**
   ```bash
   cd stock-screener
   uv run streamlit run src/stock_screener/app.py
   ```

2. **Or use the helper script:**
   ```bash
   cd stock-screener
   uv run python scripts.py app
   ```

3. **Install in development mode:**
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

- `src/stock_screener/data_provider.py`: Handles data fetching from NSE
- `src/stock_screener/strategy.py`: Contains screening strategies and strategy manager
- `src/stock_screener/screener.py`: Main orchestrator that combines data and strategies
- `src/stock_screener/app.py`: Streamlit frontend

## Adding New Strategies

To add a new strategy, create a class inheriting from `Strategy` in `strategy.py`:

```python
class MyCustomStrategy(Strategy):
    def __init__(self):
        super().__init__("My Strategy", "Description of my strategy")
    
    def apply(self, stock_data: Dict[str, Any]) -> bool:
        # Your strategy logic here
        return True  # or False
```

Then register it in `StrategyManager._register_default_strategies()`.

## Note

The demo version processes only the first 10 stocks from each universe for faster execution. Remove the slice `[:10]` in `screener.py` to process all stocks.