"""
Backtesting module for StatArbX - Comprehensive backtesting using Backtrader.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from .strategy import PairsTradingStrategy, PairsTradingData
from .data import DataDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPairsTradingStrategy(PairsTradingStrategy):
    """Enhanced pairs trading strategy with additional risk controls."""
    
    params = (
        # Existing parameters
        ('entry_threshold', 2.0),
        ('exit_threshold', 0.0),
        ('stop_loss', 4.0),
        ('lookback_period', 20),
        ('position_size', 0.1),
        ('max_positions', 5),
        ('min_trade_distance', 5),
        
        # Enhanced risk management parameters
        ('max_position_value', 0.2),      # Max position value as fraction of portfolio
        ('max_daily_loss', 0.05),         # Max daily loss as fraction of portfolio
        ('min_cash_reserve', 0.1),        # Min cash reserve to maintain
        ('volatility_adjustment', True),   # Adjust position size based on volatility
        ('correlation_threshold', 0.7),    # Min correlation required for trading
        
        # Pair-specific parameters (will be updated for each pair)
        ('hedge_ratios', {}),             # Dict of hedge ratios for each pair
        ('pair_names', []),               # List of pair names
    )
    
    def __init__(self):
        super().__init__()
        self.daily_pnl = []
        self.daily_returns = []
        self.portfolio_values = []
        self.drawdown_series = []
        self.max_portfolio_value = 0
        self.trades_log = []
        
        # Update pairs with hedge ratios from params
        for i, pair in enumerate(self.pairs):
            if i < len(self.params.pair_names):
                pair_name = self.params.pair_names[i]
                pair['name'] = pair_name
                if pair_name in self.params.hedge_ratios:
                    pair['hedge_ratio'] = self.params.hedge_ratios[pair_name]
        
        logger.info(f"Enhanced strategy initialized with {len(self.pairs)} pairs")
    
    def calculate_portfolio_risk_metrics(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics."""
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        # Update tracking
        self.portfolio_values.append(portfolio_value)
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Calculate drawdown
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        else:
            current_drawdown = 0
        
        self.drawdown_series.append(current_drawdown)
        
        return {
            'portfolio_value': portfolio_value,
            'cash': cash,
            'invested': portfolio_value - cash,
            'cash_ratio': cash / portfolio_value if portfolio_value > 0 else 1,
            'current_drawdown': current_drawdown,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0
        }
    
    def check_risk_limits(self) -> bool:
        """Check if current positions comply with risk limits."""
        risk_metrics = self.calculate_portfolio_risk_metrics()
        
        # Check cash reserve
        if risk_metrics['cash_ratio'] < self.params.min_cash_reserve:
            logger.warning(f"Cash reserve below limit: {risk_metrics['cash_ratio']:.2%}")
            return False
        
        # Check daily loss limit
        if len(self.portfolio_values) >= 2:
            daily_return = (risk_metrics['portfolio_value'] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            if daily_return < -self.params.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {daily_return:.2%}")
                return False
        
        return True
    
    def adjust_position_size_for_volatility(self, pair: Dict, base_size: Tuple[int, int]) -> Tuple[int, int]:
        """Adjust position size based on recent volatility."""
        if not self.params.volatility_adjustment:
            return base_size
        
        try:
            data1, data2 = pair['data1'], pair['data2']
            
            # Calculate recent volatility (20-day)
            returns1 = []
            returns2 = []
            
            for i in range(1, min(21, len(data1))):
                ret1 = (data1.close[-i+1] - data1.close[-i]) / data1.close[-i]
                ret2 = (data2.close[-i+1] - data2.close[-i]) / data2.close[-i]
                returns1.append(ret1)
                returns2.append(ret2)
            
            if len(returns1) < 5:
                return base_size
            
            vol1 = np.std(returns1) * np.sqrt(252)  # Annualized volatility
            vol2 = np.std(returns2) * np.sqrt(252)
            
            # Average volatility
            avg_vol = (vol1 + vol2) / 2
            
            # Scale down position if volatility is high (> 30%)
            if avg_vol > 0.3:
                scale_factor = 0.3 / avg_vol
                return int(base_size[0] * scale_factor), int(base_size[1] * scale_factor)
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error adjusting position size for volatility: {str(e)}")
            return base_size
    
    def calculate_position_size(self, price1: float, price2: float, hedge_ratio: float) -> Tuple[int, int]:
        """Enhanced position sizing with risk controls."""
        portfolio_value = self.get_portfolio_value()
        
        # Base position value
        position_value = portfolio_value * self.params.position_size
        
        # Adjust for maximum position value limit
        max_position_value = portfolio_value * self.params.max_position_value
        position_value = min(position_value, max_position_value)
        
        # Calculate base size
        unit_cost = price1 + hedge_ratio * price2
        if unit_cost <= 0:
            return 0, 0
        
        units = int(position_value / unit_cost)
        base_size = (units, int(units * hedge_ratio))
        
        return base_size
    
    def check_correlation(self, pair: Dict) -> bool:
        """Check if pair correlation is above threshold."""
        try:
            data1, data2 = pair['data1'], pair['data2']
            
            # Get recent prices for correlation calculation
            prices1 = []
            prices2 = []
            
            for i in range(min(50, len(data1))):  # 50-day correlation
                prices1.append(data1.close[-i])
                prices2.append(data2.close[-i])
            
            if len(prices1) < 20:
                return True  # Not enough data, assume OK
            
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            
            return abs(correlation) >= self.params.correlation_threshold
            
        except Exception as e:
            logger.error(f"Error checking correlation: {str(e)}")
            return True
    
    def enter_long_spread(self, pair: Dict, spread: float, z_score: float):
        """Enhanced long spread entry with additional checks."""
        # Check risk limits
        if not self.check_risk_limits():
            return
        
        # Check correlation
        if not self.check_correlation(pair):
            logger.info(f"Skipping trade for {pair['name']}: correlation below threshold")
            return
        
        # Proceed with original logic
        super().enter_long_spread(pair, spread, z_score)
    
    def enter_short_spread(self, pair: Dict, spread: float, z_score: float):
        """Enhanced short spread entry with additional checks."""
        # Check risk limits
        if not self.check_risk_limits():
            return
        
        # Check correlation
        if not self.check_correlation(pair):
            logger.info(f"Skipping trade for {pair['name']}: correlation below threshold")
            return
        
        # Proceed with original logic
        super().enter_short_spread(pair, spread, z_score)
    
    def notify_trade(self, trade):
        """Enhanced trade notification with logging."""
        if not trade.isclosed:
            return
        
        # Log trade details
        trade_info = {
            'date': self.datas[0].datetime.date(0).isoformat(),
            'pnl_gross': trade.pnl,
            'pnl_net': trade.pnlcomm,
            'commission': trade.commission,
            'data_name': trade.data._name if hasattr(trade.data, '_name') else 'Unknown'
        }
        
        self.trades_log.append(trade_info)
        super().notify_trade(trade)


class RiskManager:
    """Risk management utilities for backtesting."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = RiskManager.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_maximum_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""
        if len(equity_curve) == 0:
            return {'max_drawdown': 0, 'drawdown_duration': 0, 'recovery_time': 0}
        
        # Calculate running maximum
        peak = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - peak) / peak
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
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
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_duration': max_dd_duration,
            'drawdown_series': drawdown
        }


class StatArbBacktester:
    """Main backtesting class for statistical arbitrage strategies."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital for backtest
        """
        self.initial_capital = initial_capital
        self.cerebro = None
        self.results = None
        self.strategy_instance = None
    
    def setup_cerebro(self, commission: float = 0.001, slippage: float = 0.0005) -> bt.Cerebro:
        """
        Set up Backtrader cerebro with broker settings.
        
        Args:
            commission: Commission rate (default 0.1%)
            slippage: Slippage rate (default 0.05%)
            
        Returns:
            Configured cerebro instance
        """
        cerebro = bt.Cerebro()
        
        # Set initial capital
        cerebro.broker.setcash(self.initial_capital)
        
        # Set commission
        cerebro.broker.setcommission(
            commission=commission,
            margin=None,
            mult=1.0,
            commtype=bt.CommInfoBase.COMM_PERC,
            percabs=True,
            stocklike=True
        )
        
        # Add slippage if specified
        if slippage > 0:
            cerebro.broker.set_slippage_perc(perc=slippage)
        
        return cerebro
    
    def add_data_feeds(self, cerebro: bt.Cerebro, price_data: pd.DataFrame, 
                       pairs_info: List[Dict]) -> bt.Cerebro:
        """
        Add data feeds to cerebro for all pairs.
        
        Args:
            cerebro: Cerebro instance
            price_data: Price data DataFrame
            pairs_info: List of pair information
            
        Returns:
            Cerebro with data feeds added
        """
        # Prepare data feeds for all pairs
        all_feeds = PairsTradingData.prepare_multiple_pairs(price_data, pairs_info)
        
        # Add each feed to cerebro
        for feed in all_feeds:
            cerebro.adddata(feed)
        
        return cerebro
    
    def run_backtest(self, price_data: pd.DataFrame, pairs_info: List[Dict], 
                     strategy_params: Dict = None, 
                     commission: float = 0.001,
                     slippage: float = 0.0005) -> Dict[str, Any]:
        """
        Run complete backtest.
        
        Args:
            price_data: Price data DataFrame
            pairs_info: List of pair information
            strategy_params: Strategy parameters
            commission: Commission rate
            slippage: Slippage rate
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Setting up backtest...")
        
        # Setup cerebro
        self.cerebro = self.setup_cerebro(commission, slippage)
        
        # Add data feeds
        self.cerebro = self.add_data_feeds(self.cerebro, price_data, pairs_info)
        
        # Prepare strategy parameters
        if strategy_params is None:
            strategy_params = {}
        
        # Add pair-specific information to strategy params
        hedge_ratios = {pair['name']: pair['hedge_ratio'] for pair in pairs_info}
        pair_names = [pair['name'] for pair in pairs_info]
        
        strategy_params.update({
            'hedge_ratios': hedge_ratios,
            'pair_names': pair_names
        })
        
        # Add strategy
        self.cerebro.addstrategy(EnhancedPairsTradingStrategy, **strategy_params)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.CalmarRatio, _name='calmar')
        
        logger.info("Running backtest...")
        
        # Run backtest
        self.results = self.cerebro.run()
        
        # Extract strategy instance
        self.strategy_instance = self.results[0]
        
        logger.info("Backtest completed.")
        
        # Process results
        return self.process_results()
    
    def process_results(self) -> Dict[str, Any]:
        """Process and return backtest results."""
        if not self.results or not self.strategy_instance:
            raise ValueError("No results to process. Run backtest first.")
        
        # Extract analyzer results
        analyzers = {}
        for analyzer_name in ['sharpe', 'drawdown', 'returns', 'trades', 'sqn', 'calmar']:
            analyzer = getattr(self.strategy_instance.analyzers, analyzer_name, None)
            if analyzer:
                analyzers[analyzer_name] = analyzer.get_analysis()
        
        # Calculate additional metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Get portfolio values from strategy
        portfolio_values = getattr(self.strategy_instance, 'portfolio_values', [])
        
        if portfolio_values:
            returns_series = pd.Series([
                (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                for i in range(1, len(portfolio_values))
            ])
            
            # Calculate additional risk metrics
            risk_metrics = {
                'var_95': RiskManager.calculate_var(returns_series, 0.05),
                'var_99': RiskManager.calculate_var(returns_series, 0.01),
                'expected_shortfall': RiskManager.calculate_expected_shortfall(returns_series, 0.05),
                'volatility': returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
            }
        else:
            risk_metrics = {}
        
        # Compile results
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'portfolio_values': portfolio_values,
            'analyzers': analyzers,
            'risk_metrics': risk_metrics,
            'trades_log': getattr(self.strategy_instance, 'trades_log', []),
            'strategy_instance': self.strategy_instance
        }
        
        return results
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from backtest results."""
        if hasattr(self.strategy_instance, 'portfolio_values'):
            # Create date index if possible
            try:
                dates = pd.date_range(
                    start=self.strategy_instance.datas[0].datetime.date(-len(self.strategy_instance.portfolio_values)+1),
                    periods=len(self.strategy_instance.portfolio_values),
                    freq='D'
                )
                return pd.Series(self.strategy_instance.portfolio_values, index=dates)
            except:
                return pd.Series(self.strategy_instance.portfolio_values)
        else:
            return pd.Series([])


if __name__ == "__main__":
    # Example usage
    from .cointegration import CointegrationAnalyzer
    
    # Download sample data
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    downloader = DataDownloader(tickers, "2020-01-01", "2023-12-31")
    data = downloader.download_data()
    
    # Find cointegrated pairs
    analyzer = CointegrationAnalyzer(data)
    cointegrated_pairs = analyzer.find_cointegrated_pairs()
    
    if cointegrated_pairs:
        # Prepare pairs info for backtesting
        pairs_info = []
        for i, pair in enumerate(cointegrated_pairs[:2]):  # Test with first 2 pairs
            pairs_info.append({
                'name': f"{pair['ticker1']}-{pair['ticker2']}",
                'ticker1': pair['ticker1'],
                'ticker2': pair['ticker2'],
                'hedge_ratio': pair['hedge_ratio']
            })
        
        # Run backtest
        backtester = StatArbBacktester(initial_capital=100000)
        
        strategy_params = {
            'entry_threshold': 2.0,
            'exit_threshold': 0.0,
            'stop_loss': 3.0,
            'position_size': 0.1
        }
        
        results = backtester.run_backtest(
            price_data=data,
            pairs_info=pairs_info,
            strategy_params=strategy_params
        )
        
        # Print results
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Final Value: ${results['final_value']:,.2f}")
        
        if 'sharpe' in results['analyzers']:
            sharpe = results['analyzers']['sharpe'].get('sharperatio', 'N/A')
            print(f"Sharpe Ratio: {sharpe}")
        
        if 'drawdown' in results['analyzers']:
            max_dd = results['analyzers']['drawdown'].get('max', {}).get('drawdown', 'N/A')
            print(f"Max Drawdown: {max_dd}%")
    
    else:
        print("No cointegrated pairs found for backtesting.")
