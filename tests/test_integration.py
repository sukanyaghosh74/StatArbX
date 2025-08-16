"""
Integration tests for StatArbX.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from statarbx.data import DataDownloader
from statarbx.cointegration import CointegrationAnalyzer
from statarbx.metrics import PerformanceMetrics


@pytest.fixture
def sample_correlated_data():
    """Generate sample correlated stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-06-30', freq='D')
    
    # Generate correlated returns
    n = len(dates)
    returns1 = np.random.normal(0.001, 0.02, n)
    returns2 = 0.7 * returns1 + 0.5 * np.random.normal(0.001, 0.02, n)
    
    # Convert to prices
    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 200 * np.cumprod(1 + returns2)
    
    return pd.DataFrame({
        'AAPL': prices1,
        'MSFT': prices2
    }, index=dates)


def test_data_to_cointegration_pipeline(sample_correlated_data):
    """Test the pipeline from data loading to cointegration analysis."""
    # Simulate data loading
    downloader = DataDownloader(['AAPL', 'MSFT'], '2020-01-01', '2020-06-30')
    downloader.data = sample_correlated_data
    
    # Test data quality
    quality = downloader.check_data_quality()
    assert len(quality) == 2
    assert all(q['missing_values'] == 0 for q in quality.values())
    
    # Test cointegration analysis
    analyzer = CointegrationAnalyzer(sample_correlated_data)
    pairs = analyzer.find_cointegrated_pairs()
    
    # Should find at least one pair (or none, depending on random data)
    assert isinstance(pairs, list)
    
    if pairs:
        # Test first pair
        pair = pairs[0]
        assert 'ticker1' in pair
        assert 'ticker2' in pair
        assert 'hedge_ratio' in pair
        assert 'p_value' in pair
        assert 'half_life' in pair
        assert 'spread' in pair


def test_full_metrics_pipeline(sample_correlated_data):
    """Test full pipeline ending with metrics calculation."""
    # Mock equity curve
    dates = sample_correlated_data.index
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.01, len(dates))
    equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
    
    # Mock trades
    trades = [150, -80, 220, -45, 180, -60, 320, -95, 200, -30]
    
    # Calculate comprehensive metrics
    metrics = PerformanceMetrics.calculate_comprehensive_metrics(
        equity_curve, trades
    )
    
    # Verify metrics are reasonable
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    
    # Check ranges
    assert -1 <= metrics['max_drawdown'] <= 0
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['profit_factor'] >= 0
    assert metrics['total_trades'] == len(trades)


def test_package_imports():
    """Test that all main package components can be imported."""
    from statarbx import (
        DataDownloader, CointegrationAnalyzer, StatArbBacktester,
        PerformanceMetrics, Visualizer, ResultsExporter,
        __version__
    )
    
    # Check version exists
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    
    # Check classes can be instantiated
    downloader = DataDownloader(['TEST'], '2020-01-01', '2020-12-31')
    assert downloader is not None
    
    analyzer = CointegrationAnalyzer(pd.DataFrame())
    assert analyzer is not None
    
    backtester = StatArbBacktester()
    assert backtester is not None
    
    visualizer = Visualizer()
    assert visualizer is not None


@patch('statarbx.data.yf.download')
def test_data_download_error_handling(mock_download):
    """Test error handling in data download."""
    # Mock download to raise an exception
    mock_download.side_effect = Exception("Network error")
    
    downloader = DataDownloader(['AAPL'], '2020-01-01', '2020-12-31')
    
    with pytest.raises(Exception):
        downloader.download_data()


def test_cointegration_with_insufficient_data():
    """Test cointegration analysis with insufficient data."""
    # Create very small dataset
    dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    data = pd.DataFrame({
        'A': [100, 101, 102, 103, 104],
        'B': [200, 201, 202, 203, 204]
    }, index=dates)
    
    analyzer = CointegrationAnalyzer(data)
    # Should handle gracefully or return empty results
    pairs = analyzer.find_cointegrated_pairs()
    
    assert isinstance(pairs, list)
    # May be empty due to insufficient data


def test_metrics_with_edge_cases():
    """Test metrics calculation with edge cases."""
    # Test with flat equity curve
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    flat_curve = pd.Series([100] * len(dates), index=dates)
    
    metrics = PerformanceMetrics.calculate_comprehensive_metrics(flat_curve, [])
    
    assert metrics['total_return'] == 0
    assert metrics['volatility'] == 0
    assert metrics['cagr'] == 0
    assert metrics['max_drawdown'] == 0
    
    # Test with empty trades
    assert metrics['total_trades'] == 0
    assert metrics['win_rate'] == 0
    assert metrics['profit_factor'] == 0


def test_config_integration():
    """Test configuration system integration."""
    from statarbx.config import StatArbConfig, get_config
    
    # Test default config
    config = StatArbConfig()
    assert config.strategy.entry_threshold == 2.0
    assert config.backtest.initial_capital == 100000.0
    
    # Test config to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert 'strategy' in config_dict
    assert 'backtest' in config_dict
