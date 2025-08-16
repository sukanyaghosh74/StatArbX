"""
StatArbX - Statistical Arbitrage Trading Bot
============================================

A comprehensive statistical arbitrage (pairs trading) bot that downloads stock data 
from Yahoo Finance, finds cointegrated pairs using the Engle-Granger test, and 
implements a z-score based trading strategy with advanced backtesting capabilities.

Features:
---------
- Data download from Yahoo Finance using yfinance
- Cointegration analysis with Engle-Granger test
- Hedge ratio and half-life calculation
- Z-score based trading strategy with entry/exit rules
- Comprehensive backtesting using Backtrader
- Risk management with commissions, slippage, and position sizing
- Performance metrics (Sharpe, Sortino, CAGR, Max Drawdown, etc.)
- Visualization with equity curves and spread analysis
- CLI interface for easy execution
- Professional project structure with testing support

Example Usage:
--------------
```python
from statarbx import DataDownloader, CointegrationAnalyzer, StatArbBacktester

# Download data
downloader = DataDownloader(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
data = downloader.download_data()

# Find cointegrated pairs
analyzer = CointegrationAnalyzer(data)
pairs = analyzer.find_cointegrated_pairs()

# Run backtest
backtester = StatArbBacktester(initial_capital=100000)
results = backtester.run_backtest(data, pairs)
```

CLI Usage:
----------
```bash
python scripts/run_backtest.py --tickers "AAPL,MSFT,GOOGL,AMZN" --start 2018-01-01 --end 2024-12-31
```

Author: StatArbX Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "StatArbX Team"
__email__ = "contact@statarbx.com"
__license__ = "MIT"

# Core imports for easy access
from .data import DataDownloader, DataPreprocessor
from .cointegration import CointegrationAnalyzer, SpreadAnalyzer
from .strategy import PairsTradingStrategy, SignalGenerator, PairsTradingData
from .backtester import StatArbBacktester, EnhancedPairsTradingStrategy, RiskManager
from .metrics import PerformanceMetrics, Visualizer, ResultsExporter

# Version and metadata
__all__ = [
    # Core classes
    'DataDownloader',
    'DataPreprocessor', 
    'CointegrationAnalyzer',
    'SpreadAnalyzer',
    'PairsTradingStrategy',
    'EnhancedPairsTradingStrategy',
    'SignalGenerator',
    'PairsTradingData',
    'StatArbBacktester',
    'RiskManager',
    'PerformanceMetrics',
    'Visualizer',
    'ResultsExporter',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

# Package information
PACKAGE_INFO = {
    'name': 'StatArbX',
    'version': __version__,
    'description': 'Statistical Arbitrage Trading Bot with Pairs Trading Strategy',
    'author': __author__,
    'email': __email__,
    'license': __license__,
    'url': 'https://github.com/statarbx/statarbx',
    'keywords': ['trading', 'quantitative', 'statistical-arbitrage', 'pairs-trading', 'backtrader'],
}

def get_version():
    """Return the current version of StatArbX."""
    return __version__

def get_info():
    """Return package information."""
    return PACKAGE_INFO.copy()

# Configure package-level logging
import logging

# Set up package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add null handler to prevent "No handler found" warnings
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Optional: Add console handler for development
def setup_logging(level=logging.INFO, format_str=None):
    """
    Set up logging for StatArbX package.
    
    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string (optional)
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set package logger level
    logger.setLevel(level)
    
    logger.info(f"StatArbX v{__version__} logging initialized")

# Suppress warnings from dependencies (optional)
import warnings

def suppress_warnings():
    """Suppress common warnings from dependencies."""
    warnings.filterwarnings('ignore', category=UserWarning, module='yfinance')
    warnings.filterwarnings('ignore', category=FutureWarning, module='backtrader')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

# Optional: Auto-suppress warnings in production
import os
if os.environ.get('STATARBX_SUPPRESS_WARNINGS', '').lower() in ('true', '1', 'yes'):
    suppress_warnings()

# Health check function
def health_check():
    """
    Perform a basic health check of the StatArbX package.
    
    Returns:
        dict: Health check results
    """
    results = {
        'version': __version__,
        'imports': {},
        'dependencies': {},
        'status': 'healthy'
    }
    
    # Test core imports
    core_modules = [
        'data', 'cointegration', 'strategy', 
        'backtester', 'metrics'
    ]
    
    for module in core_modules:
        try:
            __import__(f'statarbx.{module}')
            results['imports'][module] = 'OK'
        except ImportError as e:
            results['imports'][module] = f'ERROR: {str(e)}'
            results['status'] = 'degraded'
    
    # Test key dependencies
    key_deps = [
        'pandas', 'numpy', 'matplotlib', 'scipy',
        'yfinance', 'backtrader', 'statsmodels'
    ]
    
    for dep in key_deps:
        try:
            __import__(dep)
            results['dependencies'][dep] = 'OK'
        except ImportError as e:
            results['dependencies'][dep] = f'ERROR: {str(e)}'
            results['status'] = 'degraded'
    
    return results

# Banner for CLI
def print_banner():
    """Print StatArbX banner."""
    banner = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                          StatArbX                            ║
    ║              Statistical Arbitrage Trading Bot              ║
    ║                        Version {__version__}                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    A comprehensive pairs trading system with:
    • Cointegration analysis (Engle-Granger test)
    • Z-score based trading strategy
    • Advanced backtesting with Backtrader
    • Risk management and performance metrics
    • Professional visualization and reporting
    
    Author: {__author__}
    License: {__license__}
    """
    print(banner)
