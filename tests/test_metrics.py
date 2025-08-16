"""
Test suite for the metrics module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock
import tempfile
import os

from statarbx.metrics import PerformanceMetrics, Visualizer, ResultsExporter


class TestPerformanceMetrics:
    """Test class for PerformanceMetrics."""
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        equity_curve = pd.Series([100, 102, 104, 103, 105], index=dates)
        
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == 4  # One less than original
        assert np.allclose(returns.iloc[0], 0.02)  # (102-100)/100
        assert np.allclose(returns.iloc[-1], (105-103)/103)
    
    def test_calculate_cagr(self):
        """Test CAGR calculation."""
        dates = pd.date_range('2020-01-01', '2022-01-01', freq='D')
        # 2 years, 10% total return
        equity_curve = pd.Series([100] + [110] * (len(dates)-1), index=dates)
        
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        
        # Should be approximately (1.1)^(1/2) - 1 ≈ 0.0488
        assert 0.04 < cagr < 0.05
    
    def test_calculate_cagr_empty(self):
        """Test CAGR calculation with empty series."""
        equity_curve = pd.Series([])
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        assert cagr == 0.0
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))  # Daily returns for 1 year
        
        vol_annual = PerformanceMetrics.calculate_volatility(returns, annualize=True)
        vol_daily = PerformanceMetrics.calculate_volatility(returns, annualize=False)
        
        # Annualized should be roughly sqrt(252) times daily
        assert np.abs(vol_annual - vol_daily * np.sqrt(252)) < 0.01
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        np.random.seed(42)
        # Create returns with positive expected return
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        # Sortino should generally be higher than Sharpe for same returns
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        equity_curve = pd.Series([100, 105, 110, 108, 106, 104, 102, 106, 108, 112], index=dates)
        
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        assert 'max_drawdown' in dd_metrics
        assert 'max_drawdown_pct' in dd_metrics
        assert 'drawdown_duration' in dd_metrics
        
        # Max drawdown should be from peak at 110 to trough at 102
        expected_dd_pct = (102 - 110) / 110  # -0.0727...
        assert np.abs(dd_metrics['max_drawdown_pct'] - abs(expected_dd_pct)) < 0.001
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        cagr = 0.1  # 10% CAGR
        max_drawdown = 0.05  # 5% max drawdown
        
        calmar = PerformanceMetrics.calculate_calmar_ratio(cagr, max_drawdown)
        
        assert calmar == 2.0  # 0.1 / 0.05
    
    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        trades_pnl = [100, -50, 200, -25, 150, -75, 300]  # 4 wins, 3 losses
        
        win_rate = PerformanceMetrics.calculate_win_rate(trades_pnl)
        
        assert win_rate == 4/7  # 57.14%
    
    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        trades_pnl = [100, -50, 200, -25]  # Gross profit: 300, Gross loss: 75
        
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades_pnl)
        
        assert profit_factor == 300/75  # 4.0
    
    def test_calculate_var(self):
        """Test VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = PerformanceMetrics.calculate_var(returns, confidence_level=0.05)
        
        # Should be approximately the 5th percentile
        assert var_95 < 0  # Should be negative (loss)
        assert np.abs(var_95 - np.percentile(returns, 5)) < 0.001
    
    def test_calculate_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        es_95 = PerformanceMetrics.calculate_expected_shortfall(returns, confidence_level=0.05)
        
        # ES should be more negative than VaR
        var_95 = PerformanceMetrics.calculate_var(returns, confidence_level=0.05)
        assert es_95 < var_95
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        
        trades_pnl = [100, -50, 200, -25, 150, -75, 300, -100]
        
        metrics = PerformanceMetrics.calculate_comprehensive_metrics(
            equity_curve, trades_pnl
        )
        
        # Check that all expected metrics are present
        expected_keys = [
            'total_return', 'cagr', 'volatility', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'var_95', 'var_99', 'expected_shortfall_95',
            'win_rate', 'profit_factor', 'total_trades', 'winning_trades', 'losing_trades'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


class TestVisualizer:
    """Test class for Visualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        self.equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        
        self.spread = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        self.z_score = (self.spread - self.spread.rolling(20).mean()) / self.spread.rolling(20).std()
    
    def test_visualizer_init(self):
        """Test Visualizer initialization."""
        viz = Visualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_equity_curve(self, mock_show):
        """Test equity curve plotting."""
        self.setUp()
        
        fig = self.visualizer.plot_equity_curve(self.equity_curve)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    @patch('matplotlib.pyplot.show')
    def test_plot_drawdown(self, mock_show):
        """Test drawdown plotting."""
        self.setUp()
        
        fig = self.visualizer.plot_drawdown(self.equity_curve)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_spread_analysis(self, mock_show):
        """Test spread analysis plotting."""
        self.setUp()
        
        fig = self.visualizer.plot_spread_analysis(self.spread, self.z_score)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_returns_distribution(self, mock_show):
        """Test returns distribution plotting."""
        self.setUp()
        
        returns = self.equity_curve.pct_change().dropna()
        fig = self.visualizer.plot_returns_distribution(returns)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_rolling_metrics(self, mock_show):
        """Test rolling metrics plotting."""
        self.setUp()
        
        fig = self.visualizer.plot_rolling_metrics(self.equity_curve, window=50)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestResultsExporter:
    """Test class for ResultsExporter."""
    
    def test_export_metrics_to_json(self):
        """Test JSON export functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            metrics = {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.05,
                'win_rate': 0.6,
                'test_array': np.array([1, 2, 3]),
                'test_series': pd.Series([1, 2, 3])
            }
            
            ResultsExporter.export_metrics_to_json(metrics, temp_file)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r') as f:
                import json
                loaded_data = json.load(f)
                
            assert 'total_return' in loaded_data
            assert loaded_data['total_return'] == 0.15
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_equity_curve_to_csv(self):
        """Test CSV export functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
            equity_curve = pd.Series([100, 102, 104, 103, 105], index=dates)
            
            ResultsExporter.export_equity_curve_to_csv(equity_curve, temp_file)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_file)
            
            loaded_df = pd.read_csv(temp_file)
            assert len(loaded_df) == 5
            assert 'date' in loaded_df.columns
            assert 'portfolio_value' in loaded_df.columns
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_create_performance_report(self):
        """Test performance report creation."""
        metrics = {
            'total_return': 0.15,
            'cagr': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'calmar_ratio': 2.0,
            'max_drawdown': 0.05,
            'var_95': -0.02,
            'var_99': -0.03,
            'expected_shortfall_95': -0.025,
            'total_trades': 50,
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'winning_trades': 30,
            'losing_trades': 20,
            'beta': 0.8,
            'alpha': 0.02,
            'information_ratio': 1.1
        }
        
        report = ResultsExporter.create_performance_report(metrics)
        
        assert isinstance(report, str)
        assert 'STATISTICAL ARBITRAGE PERFORMANCE REPORT' in report
        assert 'Total Return:' in report
        assert 'Sharpe Ratio:' in report
        assert 'Max Drawdown:' in report


@pytest.fixture
def sample_equity_curve():
    """Fixture providing sample equity curve."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    return pd.Series((1 + returns).cumprod() * 100000, index=dates)


@pytest.fixture
def sample_trades():
    """Fixture providing sample trades P&L."""
    return [100, -50, 200, -25, 150, -75, 300, -100, 180, -40]


def test_metrics_integration(sample_equity_curve, sample_trades):
    """Integration test for metrics calculation."""
    # Test comprehensive metrics calculation
    metrics = PerformanceMetrics.calculate_comprehensive_metrics(
        sample_equity_curve, sample_trades
    )
    
    # Verify all metrics are reasonable
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['total_trades'] == len(sample_trades)
    assert metrics['profit_factor'] > 0
    assert -1 <= metrics['max_drawdown'] <= 0
    
    # Test visualization
    visualizer = Visualizer()
    
    # Should not raise exceptions
    fig1 = visualizer.plot_equity_curve(sample_equity_curve)
    fig2 = visualizer.plot_drawdown(sample_equity_curve)
    
    plt.close(fig1)
    plt.close(fig2)
    
    # Test report generation
    report = ResultsExporter.create_performance_report(metrics)
    assert len(report) > 100  # Should be a substantial report
