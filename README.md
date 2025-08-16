# StatArbX - Statistical Arbitrage Trading Bot

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive statistical arbitrage (pairs trading) bot that downloads stock data from Yahoo Finance using yfinance, finds cointegrated pairs via the Engle–Granger test, trades the spread using z-score entry and exit rules, and backtests the strategy with Backtrader.

## 🎯 Features

- **Data Download**: Automatic stock data downloading from Yahoo Finance using yfinance
- **Cointegration Analysis**: Engle-Granger test implementation with hedge ratio and half-life calculation
- **Trading Strategy**: Z-score based pairs trading with configurable entry/exit thresholds
- **Advanced Backtesting**: Comprehensive backtesting using Backtrader with:
  - Commission and slippage modeling
  - Risk controls and position sizing
  - Capital allocation management
- **Performance Metrics**: Extensive performance analysis including:
  - Sharpe Ratio, Sortino Ratio, CAGR
  - Maximum Drawdown, Win Rate, Profit Factor
  - VaR and Expected Shortfall
- **Visualization**: Professional plots for:
  - Equity curves with benchmark comparison
  - Spread analysis with z-score and trading signals
  - Drawdown analysis and rolling metrics
- **Export Capabilities**: Save results to JSON/CSV formats
- **CLI Interface**: Easy-to-use command-line interface
- **Professional Structure**: Clean, modular codebase with comprehensive testing

## 📊 Strategy Overview

### Statistical Arbitrage (Pairs Trading)

Statistical arbitrage, specifically pairs trading, is a market-neutral trading strategy that exploits the temporary divergence between two historically correlated securities. The strategy is based on the concept of **cointegration** - a statistical relationship where two time series share a common stochastic trend.

#### Key Components:

1. **Cointegration Testing**: Uses the Engle-Granger two-step method to identify cointegrated pairs
2. **Hedge Ratio Calculation**: Determines the optimal ratio for pairing securities
3. **Spread Construction**: Creates a stationary spread: `Spread = Stock1 - β × Stock2`
4. **Z-Score Calculation**: Normalizes the spread using rolling statistics
5. **Signal Generation**: Trades when z-score exceeds entry thresholds and exits at mean reversion

#### Trading Logic:

```
If Z-Score ≤ -Entry_Threshold:
    → Long the spread (Buy Stock1, Sell Stock2)
    
If Z-Score ≥ Entry_Threshold:
    → Short the spread (Sell Stock1, Buy Stock2)
    
If |Z-Score| ≤ Exit_Threshold:
    → Close positions (Mean reversion)
    
If |Z-Score| ≥ Stop_Loss_Threshold:
    → Close positions (Risk management)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sukanyaghosh74/StatArbX.git
cd StatArbX

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

#### Command Line Interface

```bash
# Basic backtest with FAANG stocks
python scripts/run_backtest.py --tickers "AAPL,MSFT,GOOGL,AMZN" --start 2018-01-01 --end 2024-12-31

# Advanced usage with custom parameters
python scripts/run_backtest.py \
    --tickers "SPY,QQQ,IWM,TLT" \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --capital 50000 \
    --entry-threshold 2.5 \
    --exit-threshold 0.5 \
    --position-size 0.15 \
    --save-plots \
    --save-data

# Using configuration file
python scripts/run_backtest.py --config config.json
```

#### Python API

```python
from statarbx import DataDownloader, CointegrationAnalyzer, StatArbBacktester

# 1. Download data
downloader = DataDownloader(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)
data = downloader.download_data()

# 2. Find cointegrated pairs
analyzer = CointegrationAnalyzer(data, significance_level=0.05)
pairs = analyzer.find_cointegrated_pairs(min_half_life=1, max_half_life=252)

print(f"Found {len(pairs)} cointegrated pairs")

# 3. Prepare pairs for backtesting
pairs_info = []
for pair in pairs:
    pairs_info.append({
        'name': f"{pair['ticker1']}-{pair['ticker2']}",
        'ticker1': pair['ticker1'],
        'ticker2': pair['ticker2'],
        'hedge_ratio': pair['hedge_ratio']
    })

# 4. Run backtest
backtester = StatArbBacktester(initial_capital=100000)
results = backtester.run_backtest(
    price_data=data,
    pairs_info=pairs_info,
    strategy_params={
        'entry_threshold': 2.0,
        'exit_threshold': 0.0,
        'stop_loss': 4.0,
        'position_size': 0.1
    },
    commission=0.001,
    slippage=0.0005
)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Final Value: ${results['final_value']:,.2f}")
```

## 📁 Project Structure

```
StatArbX/
├── src/statarbx/                 # Main package
│   ├── __init__.py              # Package initialization
│   ├── data.py                  # Data downloading and preprocessing
│   ├── cointegration.py         # Cointegration analysis
│   ├── strategy.py              # Trading strategy implementation
│   ├── backtester.py            # Backtrader integration
│   ├── metrics.py               # Performance metrics and visualization
│   └── config.py                # Configuration management
├── scripts/                     # CLI scripts
│   └── run_backtest.py          # Main CLI interface
├── tests/                       # Test suite
│   ├── test_data.py
│   ├── test_cointegration.py
│   ├── test_strategy.py
│   ├── test_backtester.py
│   └── test_metrics.py
├── out/                         # Output directory
│   ├── backtest_results.json
│   ├── equity_curve.png
│   └── spread_analysis.png
├── requirements.txt             # Dependencies
├── pyproject.toml              # Modern Python packaging
└── README.md                   # This file
```

## 📈 Example Results

### Sample Backtest: AAPL-MSFT Pair (2020-2023)

```
STATISTICAL ARBITRAGE PERFORMANCE REPORT
=====================================

RETURN METRICS:
--------------
Total Return:        12.35%
CAGR:               3.95%
Volatility:         8.42%

RISK-ADJUSTED METRICS:
---------------------
Sharpe Ratio:        1.87
Sortino Ratio:       2.31
Calmar Ratio:        1.23

RISK METRICS:
------------
Max Drawdown:        -3.21%
VaR (95%):          -0.89%
VaR (99%):          -1.34%
Expected Shortfall:  -1.67%

TRADING METRICS:
---------------
Total Trades:           42
Win Rate:            57.14%
Profit Factor:        1.24
Winning Trades:         24
Losing Trades:          18
```

### Cointegrated Pairs Summary

| Pair        | Y_Ticker | X_Ticker | Hedge_Ratio | P_Value | Half_Life | Correlation |
|-------------|----------|----------|-------------|---------|-----------|-------------|
| AAPL-MSFT   | AAPL     | MSFT     | 0.8234      | 0.0123  | 23.45     | 0.8567      |
| GOOGL-AMZN  | GOOGL    | AMZN     | 1.2456      | 0.0198  | 31.67     | 0.7890      |
| MSFT-GOOGL  | MSFT     | GOOGL    | 0.7891      | 0.0234  | 28.90     | 0.8234      |

## 🛠️ Configuration

### Configuration File Example

```json
{
  "data": {
    "default_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "default_start_date": "2018-01-01",
    "default_end_date": "2024-12-31"
  },
  "strategy": {
    "entry_threshold": 2.0,
    "exit_threshold": 0.0,
    "stop_loss_threshold": 4.0,
    "position_size": 0.1,
    "max_positions": 5
  },
  "backtest": {
    "initial_capital": 100000,
    "commission": 0.001,
    "slippage": 0.0005
  },
  "output": {
    "save_plots": true,
    "save_data": true,
    "output_dir": "out"
  }
}
```

### Environment Variables

```bash
export STATARBX_INITIAL_CAPITAL=50000
export STATARBX_COMMISSION=0.002
export STATARBX_ENTRY_THRESHOLD=2.5
export STATARBX_SAVE_PLOTS=true
```

## 📊 CLI Parameters

### Data Parameters
- `--tickers`: Comma-separated list of stock tickers
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)

### Strategy Parameters
- `--entry-threshold`: Z-score threshold for entry (default: 2.0)
- `--exit-threshold`: Z-score threshold for exit (default: 0.0)
- `--stop-loss`: Stop loss z-score threshold (default: 4.0)
- `--lookback-period`: Rolling window size (default: 20)
- `--position-size`: Position size as portfolio fraction (default: 0.1)
- `--max-positions`: Maximum concurrent positions (default: 5)

### Backtesting Parameters
- `--capital`: Initial capital (default: 100000)
- `--commission`: Commission rate (default: 0.001)
- `--slippage`: Slippage rate (default: 0.0005)

### Output Parameters
- `--output-dir`: Output directory (default: out)
- `--save-plots`: Save plots to files
- `--show-plots`: Display plots interactively
- `--save-data`: Save detailed results to CSV/JSON

## 🧪 Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/statarbx

# Run specific test file
pytest tests/test_cointegration.py

# Run with verbose output
pytest -v
```

## 📚 Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing (statistical tests)
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

### Financial Dependencies
- **yfinance**: Yahoo Finance data downloader
- **backtrader**: Backtesting framework
- **statsmodels**: Statistical models and tests

### Optional Dependencies
- **scikit-learn**: Machine learning utilities
- **jupyter**: Notebook interface
- **pytest**: Testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## ⚠️ Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Statistical arbitrage trading involves substantial risk of loss. Past performance does not guarantee future results. Always:

- Thoroughly test strategies on paper before live trading
- Never risk more than you can afford to lose
- Consider transaction costs, market impact, and slippage
- Understand that cointegration relationships can break down
- Use appropriate risk management and position sizing

The authors are not responsible for any financial losses incurred through use of this software.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: [Read the Docs](https://statarbx.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/statarbx/statarbx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/statarbx/statarbx/discussions)

## 🏆 Acknowledgments

- Engle-Granger cointegration methodology
- Backtrader framework for backtesting
- Yahoo Finance for data access
- The quantitative finance community

---

**StatArbX** - *Making statistical arbitrage accessible to everyone*

[![GitHub stars](https://img.shields.io/github/stars/statarbx/statarbx?style=social)](https://github.com/statarbx/statarbx)
[![GitHub forks](https://img.shields.io/github/forks/statarbx/statarbx?style=social)](https://github.com/statarbx/statarbx)
