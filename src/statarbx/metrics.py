"""
Metrics and visualization module for StatArbX - Performance metrics and plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceMetrics:
    """Class for calculating comprehensive performance metrics."""
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve."""
        return equity_curve.pct_change().dropna()
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity_curve) < 2:
            return 0.0
        
        start_value = equity_curve.iloc[0]
        end_value = equity_curve.iloc[-1]
        
        if start_value <= 0:
            return 0.0
        
        # Calculate number of years
        if hasattr(equity_curve.index, 'to_pydatetime'):
            time_diff = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        else:
            time_diff = len(equity_curve) / 252  # Assume 252 trading days per year
        
        if time_diff <= 0:
            return 0.0
        
        cagr = (end_value / start_value) ** (1 / time_diff) - 1
        return cagr
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility of returns."""
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""
        if len(equity_curve) == 0:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'drawdown_duration': 0}
        
        # Calculate running maximum
        peak = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = equity_curve - peak
        drawdown_pct = drawdown / peak
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'drawdown_duration': 0}
        
        # Calculate drawdown duration
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:
            drawdown_periods.append(len(in_drawdown) - start)
        
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_pct': abs(max_drawdown_pct),
            'drawdown_duration': max_duration,
            'drawdown_series': drawdown_pct
        }
    
    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return np.inf if cagr > 0 else 0.0
        
        return cagr / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(trades_pnl: List[float]) -> float:
        """Calculate win rate from list of trade P&L."""
        if not trades_pnl:
            return 0.0
        
        winning_trades = sum(1 for pnl in trades_pnl if pnl > 0)
        return winning_trades / len(trades_pnl)
    
    @staticmethod
    def calculate_profit_factor(trades_pnl: List[float]) -> float:
        """Calculate profit factor."""
        if not trades_pnl:
            return 0.0
        
        gross_profit = sum(pnl for pnl in trades_pnl if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in trades_pnl if pnl < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall."""
        var = PerformanceMetrics.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return tail_returns.mean()
    
    @staticmethod
    def calculate_comprehensive_metrics(equity_curve: pd.Series, trades_pnl: List[float] = None,
                                      benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if trades_pnl is None:
            trades_pnl = []
        
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        # Basic metrics
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        volatility = PerformanceMetrics.calculate_volatility(returns)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
        
        # Drawdown metrics
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        calmar = PerformanceMetrics.calculate_calmar_ratio(cagr, dd_metrics['max_drawdown_pct'])
        
        # Risk metrics
        var_95 = PerformanceMetrics.calculate_var(returns, 0.05)
        var_99 = PerformanceMetrics.calculate_var(returns, 0.01)
        es_95 = PerformanceMetrics.calculate_expected_shortfall(returns, 0.05)
        
        # Trade-based metrics
        win_rate = PerformanceMetrics.calculate_win_rate(trades_pnl)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades_pnl)
        
        # Benchmark comparison
        beta = 0.0
        alpha = 0.0
        information_ratio = 0.0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns with benchmark
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 1:
                aligned_returns = returns.loc[common_index]
                aligned_benchmark = benchmark_returns.loc[common_index]
                
                # Calculate beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                
                if benchmark_variance != 0:
                    beta = covariance / benchmark_variance
                
                # Calculate alpha
                alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
                
                # Calculate information ratio
                excess_returns = aligned_returns - aligned_benchmark
                tracking_error = excess_returns.std()
                
                if tracking_error != 0:
                    information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252)
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': dd_metrics['max_drawdown_pct'],
            'calmar_ratio': calmar,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades_pnl),
            'winning_trades': sum(1 for pnl in trades_pnl if pnl > 0),
            'losing_trades': sum(1 for pnl in trades_pnl if pnl < 0),
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio
        }
        
        return metrics


class Visualizer:
    """Class for creating performance visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_equity_curve(self, equity_curve: pd.Series, benchmark: pd.Series = None,
                         title: str = "Portfolio Equity Curve") -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            equity_curve: Portfolio equity curve
            benchmark: Optional benchmark series
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve.values, 
                label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to start at same value
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax.plot(benchmark.index, normalized_benchmark.values,
                   label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if hasattr(equity_curve.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, equity_curve: pd.Series, 
                     title: str = "Portfolio Drawdown") -> plt.Figure:
        """
        Plot drawdown series.
        
        Args:
            equity_curve: Portfolio equity curve
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        drawdown_series = dd_metrics['drawdown_series']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.fill_between(drawdown_series.index, drawdown_series.values, 0,
                       alpha=0.7, color='red', label='Drawdown')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Format x-axis dates
        if hasattr(drawdown_series.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_spread_analysis(self, spread: pd.Series, z_score: pd.Series, 
                            signals: pd.DataFrame = None,
                            title: str = "Spread and Z-Score Analysis") -> plt.Figure:
        """
        Plot spread and z-score with trading signals.
        
        Args:
            spread: Spread time series
            z_score: Z-score time series
            signals: DataFrame with trading signals
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Plot spread
        ax1.plot(spread.index, spread.values, color='blue', linewidth=1)
        ax1.set_title(f"{title} - Spread", fontsize=12, fontweight='bold')
        ax1.set_ylabel('Spread', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot z-score
        ax2.plot(z_score.index, z_score.values, color='green', linewidth=1)
        
        # Add threshold lines
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (+2)')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (-2)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Mean')
        
        # Add trading signals if provided
        if signals is not None:
            long_entries = signals[signals['long_entry']]
            short_entries = signals[signals['short_entry']]
            exits = signals[signals['exit']]
            
            if not long_entries.empty:
                ax1.scatter(long_entries.index, spread.loc[long_entries.index],
                           color='green', marker='^', s=100, alpha=0.7, label='Long Entry')
                ax2.scatter(long_entries.index, z_score.loc[long_entries.index],
                           color='green', marker='^', s=100, alpha=0.7)
            
            if not short_entries.empty:
                ax1.scatter(short_entries.index, spread.loc[short_entries.index],
                           color='red', marker='v', s=100, alpha=0.7, label='Short Entry')
                ax2.scatter(short_entries.index, z_score.loc[short_entries.index],
                           color='red', marker='v', s=100, alpha=0.7)
            
            if not exits.empty:
                ax1.scatter(exits.index, spread.loc[exits.index],
                           color='orange', marker='x', s=100, alpha=0.7, label='Exit')
                ax2.scatter(exits.index, z_score.loc[exits.index],
                           color='orange', marker='x', s=100, alpha=0.7)
            
            ax1.legend()
        
        ax2.set_title(f"{title} - Z-Score", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Z-Score', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            if hasattr(spread.index, 'to_pydatetime'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self, returns: pd.Series,
                                title: str = "Returns Distribution") -> plt.Figure:
        """
        Plot returns distribution histogram.
        
        Args:
            returns: Returns series
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, density=True, color='blue', edgecolor='black')
        ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Daily Returns', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_rolling_metrics(self, equity_curve: pd.Series, window: int = 252,
                           title: str = "Rolling Performance Metrics") -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            equity_curve: Portfolio equity curve
            window: Rolling window size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        # Calculate rolling metrics
        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window=window).mean() / 
                         returns.rolling(window=window).std() * np.sqrt(252))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Rolling volatility
        ax1.plot(rolling_volatility.index, rolling_volatility.values, color='red', linewidth=2)
        ax1.set_title('Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Volatility', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Sharpe Ratio', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            if hasattr(equity_curve.index, 'to_pydatetime'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


class ResultsExporter:
    """Class for exporting results to various formats."""
    
    @staticmethod
    def export_metrics_to_json(metrics_dict: Dict[str, Any], filepath: str):
        """
        Export metrics dictionary to JSON file.
        
        Args:
            metrics_dict: Dictionary containing metrics
            filepath: Output file path
        """
        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_dict = convert_types(metrics_dict)
        
        with open(filepath, 'w') as f:
            json.dump(converted_dict, f, indent=2, default=str)
    
    @staticmethod
    def export_equity_curve_to_csv(equity_curve: pd.Series, filepath: str):
        """
        Export equity curve to CSV file.
        
        Args:
            equity_curve: Equity curve series
            filepath: Output file path
        """
        df = pd.DataFrame({'date': equity_curve.index, 'portfolio_value': equity_curve.values})
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def create_performance_report(metrics: Dict[str, float]) -> str:
        """
        Create a formatted performance report.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Formatted report string
        """
        report = """
STATISTICAL ARBITRAGE PERFORMANCE REPORT
=====================================

RETURN METRICS:
--------------
Total Return:        {total_return:>10.2%}
CAGR:               {cagr:>10.2%}
Volatility:         {volatility:>10.2%}

RISK-ADJUSTED METRICS:
---------------------
Sharpe Ratio:       {sharpe_ratio:>10.2f}
Sortino Ratio:      {sortino_ratio:>10.2f}
Calmar Ratio:       {calmar_ratio:>10.2f}

RISK METRICS:
------------
Max Drawdown:       {max_drawdown:>10.2%}
VaR (95%):          {var_95:>10.2%}
VaR (99%):          {var_99:>10.2%}
Expected Shortfall: {expected_shortfall_95:>10.2%}

TRADING METRICS:
---------------
Total Trades:       {total_trades:>10.0f}
Win Rate:           {win_rate:>10.2%}
Profit Factor:      {profit_factor:>10.2f}
Winning Trades:     {winning_trades:>10.0f}
Losing Trades:      {losing_trades:>10.0f}

BENCHMARK COMPARISON:
-------------------
Beta:               {beta:>10.2f}
Alpha:              {alpha:>10.4f}
Information Ratio:  {information_ratio:>10.2f}
        """.format(**metrics)
        
        return report.strip()


if __name__ == "__main__":
    # Example usage - create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate sample equity curve
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
    
    # Calculate metrics
    metrics_calc = PerformanceMetrics()
    sample_trades = [100, -50, 200, -25, 150, -75, 300, -100]
    
    comprehensive_metrics = metrics_calc.calculate_comprehensive_metrics(
        equity_curve, sample_trades
    )
    
    print("Sample Performance Metrics:")
    for key, value in comprehensive_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Create visualizations
    visualizer = Visualizer()
    
    # Equity curve plot
    fig1 = visualizer.plot_equity_curve(equity_curve, title="Sample Strategy Performance")
    plt.show()
    
    # Generate performance report
    exporter = ResultsExporter()
    report = exporter.create_performance_report(comprehensive_metrics)
    print("\n" + report)
