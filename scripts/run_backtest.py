#!/usr/bin/env python3
"""
CLI script for running StatArbX backtests.
Usage: python run_backtest.py --tickers "AAPL,MSFT,GOOGL,AMZN" --start 2018-01-01 --end 2024-12-31
"""

import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statarbx.data import DataDownloader
from statarbx.cointegration import CointegrationAnalyzer
from statarbx.backtester import StatArbBacktester
from statarbx.metrics import PerformanceMetrics, Visualizer, ResultsExporter
from statarbx.strategy import SignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatArbCLI:
    """Command line interface for StatArbX backtesting."""
    
    def __init__(self):
        self.args = None
        self.results = None
        
    def parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='StatArbX - Statistical Arbitrage Backtesting System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python run_backtest.py --tickers "AAPL,MSFT,GOOGL,AMZN" --start 2018-01-01 --end 2024-12-31
  python run_backtest.py --tickers "SPY,QQQ,IWM" --start 2020-01-01 --end 2023-12-31 --capital 50000
  python run_backtest.py --config config.json
            """
        )
        
        # Data parameters
        parser.add_argument('--tickers', type=str, required=False,
                           help='Comma-separated list of stock tickers (e.g., "AAPL,MSFT,GOOGL")')
        parser.add_argument('--start', type=str, required=False,
                           help='Start date in YYYY-MM-DD format')
        parser.add_argument('--end', type=str, required=False,
                           help='End date in YYYY-MM-DD format')
        
        # Strategy parameters
        parser.add_argument('--entry-threshold', type=float, default=2.0,
                           help='Z-score threshold for entry signals (default: 2.0)')
        parser.add_argument('--exit-threshold', type=float, default=0.0,
                           help='Z-score threshold for exit signals (default: 0.0)')
        parser.add_argument('--stop-loss', type=float, default=4.0,
                           help='Stop loss z-score threshold (default: 4.0)')
        parser.add_argument('--lookback-period', type=int, default=20,
                           help='Rolling window for z-score calculation (default: 20)')
        parser.add_argument('--position-size', type=float, default=0.1,
                           help='Position size as fraction of portfolio (default: 0.1)')
        parser.add_argument('--max-positions', type=int, default=5,
                           help='Maximum number of concurrent positions (default: 5)')
        
        # Backtesting parameters
        parser.add_argument('--capital', type=float, default=100000,
                           help='Initial capital for backtesting (default: 100000)')
        parser.add_argument('--commission', type=float, default=0.001,
                           help='Commission rate (default: 0.001 = 0.1%%)')
        parser.add_argument('--slippage', type=float, default=0.0005,
                           help='Slippage rate (default: 0.0005 = 0.05%%)')
        
        # Cointegration parameters
        parser.add_argument('--significance-level', type=float, default=0.05,
                           help='Significance level for cointegration test (default: 0.05)')
        parser.add_argument('--min-half-life', type=float, default=1,
                           help='Minimum acceptable half-life in days (default: 1)')
        parser.add_argument('--max-half-life', type=float, default=252,
                           help='Maximum acceptable half-life in days (default: 252)')
        
        # Output parameters
        parser.add_argument('--output-dir', type=str, default='out',
                           help='Output directory for results (default: out)')
        parser.add_argument('--save-plots', action='store_true',
                           help='Save plots to output directory')
        parser.add_argument('--show-plots', action='store_true',
                           help='Display plots interactively')
        parser.add_argument('--save-data', action='store_true',
                           help='Save detailed data to CSV/JSON files')
        
        # Configuration file
        parser.add_argument('--config', type=str,
                           help='Path to JSON configuration file')
        
        # Verbosity
        parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
        parser.add_argument('--quiet', '-q', action='store_true',
                           help='Suppress output except errors')
        
        self.args = parser.parse_args()
        
        # Load config file if provided
        if self.args.config:
            self.load_config_file(self.args.config)
        
        # Validate required arguments
        if not self.args.tickers or not self.args.start or not self.args.end:
            if not self.args.config:
                parser.error("--tickers, --start, and --end are required unless --config is provided")
        
        return self.args
    
    def load_config_file(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Override command line arguments with config values
            for key, value in config.items():
                if hasattr(self.args, key.replace('-', '_')):
                    setattr(self.args, key.replace('-', '_'), value)
                    
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging based on verbosity level."""
        if self.args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    def validate_inputs(self):
        """Validate input parameters."""
        # Validate date format
        try:
            start_date = datetime.strptime(self.args.start, '%Y-%m-%d')
            end_date = datetime.strptime(self.args.end, '%Y-%m-%d')
            
            if end_date <= start_date:
                raise ValueError("End date must be after start date")
                
        except ValueError as e:
            logger.error(f"Invalid date format: {str(e)}")
            sys.exit(1)
        
        # Validate tickers
        tickers = [ticker.strip().upper() for ticker in self.args.tickers.split(',')]
        if len(tickers) < 2:
            logger.error("At least 2 tickers are required for pairs trading")
            sys.exit(1)
        
        self.args.tickers_list = tickers
        
        # Validate numeric parameters
        if self.args.capital <= 0:
            logger.error("Capital must be positive")
            sys.exit(1)
        
        if not (0 < self.args.position_size <= 1):
            logger.error("Position size must be between 0 and 1")
            sys.exit(1)
        
        logger.info(f"Validated inputs: {len(tickers)} tickers, "
                   f"period: {self.args.start} to {self.args.end}")
    
    def download_data(self):
        """Download stock data."""
        logger.info("Downloading stock data...")
        
        downloader = DataDownloader(
            tickers=self.args.tickers_list,
            start_date=self.args.start,
            end_date=self.args.end
        )
        
        try:
            data = downloader.download_data()
            
            if data.empty:
                logger.error("No data downloaded")
                sys.exit(1)
            
            logger.info(f"Downloaded {len(data)} trading days of data")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            sys.exit(1)
    
    def find_cointegrated_pairs(self, data):
        """Find cointegrated pairs."""
        logger.info("Analyzing cointegration...")
        
        analyzer = CointegrationAnalyzer(
            price_data=data,
            significance_level=self.args.significance_level
        )
        
        try:
            pairs = analyzer.find_cointegrated_pairs(
                min_half_life=self.args.min_half_life,
                max_half_life=self.args.max_half_life
            )
            
            if not pairs:
                logger.warning("No cointegrated pairs found")
                return []
            
            logger.info(f"Found {len(pairs)} cointegrated pairs")
            
            # Display summary
            summary = analyzer.get_pair_summary(pairs)
            print("\nCointegrated Pairs Summary:")
            print("=" * 80)
            print(summary.to_string(index=False))
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error in cointegration analysis: {str(e)}")
            sys.exit(1)
    
    def run_backtest(self, data, cointegrated_pairs):
        """Run backtest on cointegrated pairs."""
        if not cointegrated_pairs:
            logger.error("No cointegrated pairs to backtest")
            sys.exit(1)
        
        logger.info("Running backtest...")
        
        # Prepare pairs info for backtesting
        pairs_info = []
        for pair in cointegrated_pairs:
            pairs_info.append({
                'name': f"{pair['ticker1']}-{pair['ticker2']}",
                'ticker1': pair['ticker1'],
                'ticker2': pair['ticker2'],
                'hedge_ratio': pair['hedge_ratio']
            })
        
        # Setup strategy parameters
        strategy_params = {
            'entry_threshold': self.args.entry_threshold,
            'exit_threshold': self.args.exit_threshold,
            'stop_loss': self.args.stop_loss,
            'lookback_period': self.args.lookback_period,
            'position_size': self.args.position_size,
            'max_positions': self.args.max_positions,
        }
        
        # Initialize backtester
        backtester = StatArbBacktester(initial_capital=self.args.capital)
        
        try:
            results = backtester.run_backtest(
                price_data=data,
                pairs_info=pairs_info,
                strategy_params=strategy_params,
                commission=self.args.commission,
                slippage=self.args.slippage
            )
            
            logger.info("Backtest completed successfully")
            return results, pairs_info, cointegrated_pairs
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            sys.exit(1)
    
    def analyze_results(self, results, pairs_info, cointegrated_pairs):
        """Analyze and display backtest results."""
        logger.info("Analyzing results...")
        
        # Calculate comprehensive metrics
        equity_curve = results.get('portfolio_values', [])
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            trades_pnl = [trade['pnl_net'] for trade in results.get('trades_log', [])]
            
            comprehensive_metrics = PerformanceMetrics.calculate_comprehensive_metrics(
                equity_series, trades_pnl
            )
        else:
            comprehensive_metrics = {}
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"Final Value:         ${results['final_value']:,.2f}")
        print(f"Total Return:        {results['total_return_pct']:.2f}%")
        
        if 'analyzers' in results:
            analyzers = results['analyzers']
            
            if 'sharpe' in analyzers and 'sharperatio' in analyzers['sharpe']:
                print(f"Sharpe Ratio:        {analyzers['sharpe']['sharperatio']:.2f}")
            
            if 'drawdown' in analyzers and 'max' in analyzers['drawdown']:
                max_dd = analyzers['drawdown']['max'].get('drawdown', 0)
                print(f"Max Drawdown:        {abs(max_dd):.2f}%")
            
            if 'trades' in analyzers:
                trade_analysis = analyzers['trades']
                total_trades = trade_analysis.get('total', {}).get('total', 0)
                won_trades = trade_analysis.get('won', {}).get('total', 0)
                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                
                print(f"Total Trades:        {total_trades}")
                print(f"Win Rate:            {win_rate:.1f}%")
        
        # Generate detailed report
        if comprehensive_metrics:
            exporter = ResultsExporter()
            report = exporter.create_performance_report(comprehensive_metrics)
            print("\n" + report)
        
        return comprehensive_metrics
    
    def create_visualizations(self, results, pairs_info, cointegrated_pairs, comprehensive_metrics):
        """Create and save/show visualizations."""
        if not (self.args.save_plots or self.args.show_plots):
            return
        
        logger.info("Creating visualizations...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            visualizer = Visualizer()
            
            # Equity curve
            equity_curve = results.get('portfolio_values', [])
            if equity_curve:
                equity_series = pd.Series(equity_curve)
                
                fig1 = visualizer.plot_equity_curve(
                    equity_series,
                    title="StatArbX Portfolio Performance"
                )
                
                if self.args.save_plots:
                    output_dir = Path(self.args.output_dir)
                    output_dir.mkdir(exist_ok=True)
                    fig1.savefig(output_dir / "equity_curve.png", dpi=300, bbox_inches='tight')
                    logger.info("Saved equity curve plot")
                
                if self.args.show_plots:
                    plt.show()
                else:
                    plt.close(fig1)
                
                # Drawdown plot
                fig2 = visualizer.plot_drawdown(equity_series)
                
                if self.args.save_plots:
                    fig2.savefig(output_dir / "drawdown.png", dpi=300, bbox_inches='tight')
                    logger.info("Saved drawdown plot")
                
                if self.args.show_plots:
                    plt.show()
                else:
                    plt.close(fig2)
            
            # Spread analysis for top pairs
            for i, pair in enumerate(cointegrated_pairs[:3]):  # Top 3 pairs
                spread = pair['spread']
                
                # Generate signals
                signal_gen = SignalGenerator()
                signals = signal_gen.generate_signals(
                    spread, 
                    entry_threshold=self.args.entry_threshold,
                    exit_threshold=self.args.exit_threshold,
                    lookback=self.args.lookback_period
                )
                
                z_score = signals['z_score']
                
                fig3 = visualizer.plot_spread_analysis(
                    spread, z_score, signals,
                    title=f"Pair Analysis: {pair['ticker1']}-{pair['ticker2']}"
                )
                
                if self.args.save_plots:
                    fig3.savefig(output_dir / f"spread_analysis_pair_{i+1}.png", 
                               dpi=300, bbox_inches='tight')
                
                if self.args.show_plots:
                    plt.show()
                else:
                    plt.close(fig3)
            
            logger.info("Visualizations completed")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def save_results(self, results, pairs_info, cointegrated_pairs, comprehensive_metrics):
        """Save detailed results to files."""
        if not self.args.save_data:
            return
        
        logger.info("Saving detailed results...")
        
        try:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            exporter = ResultsExporter()
            
            # Save comprehensive metrics
            all_results = {
                'backtest_summary': {
                    'initial_capital': results['initial_capital'],
                    'final_value': results['final_value'],
                    'total_return_pct': results['total_return_pct']
                },
                'performance_metrics': comprehensive_metrics,
                'strategy_parameters': {
                    'entry_threshold': self.args.entry_threshold,
                    'exit_threshold': self.args.exit_threshold,
                    'stop_loss': self.args.stop_loss,
                    'lookback_period': self.args.lookback_period,
                    'position_size': self.args.position_size,
                    'commission': self.args.commission,
                    'slippage': self.args.slippage
                },
                'pairs_analyzed': len(cointegrated_pairs),
                'timestamp': datetime.now().isoformat()
            }
            
            exporter.export_metrics_to_json(all_results, output_dir / "backtest_results.json")
            
            # Save equity curve
            if results.get('portfolio_values'):
                equity_series = pd.Series(results['portfolio_values'])
                exporter.export_equity_curve_to_csv(equity_series, output_dir / "equity_curve.csv")
            
            # Save pairs summary
            if cointegrated_pairs:
                analyzer = CointegrationAnalyzer(pd.DataFrame())  # Dummy analyzer for summary
                summary = analyzer.get_pair_summary(cointegrated_pairs)
                summary.to_csv(output_dir / "cointegrated_pairs.csv", index=False)
            
            # Save trades log
            if results.get('trades_log'):
                trades_df = pd.DataFrame(results['trades_log'])
                trades_df.to_csv(output_dir / "trades_log.csv", index=False)
            
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def run(self):
        """Main execution method."""
        # Parse arguments and setup
        self.parse_arguments()
        self.setup_logging()
        self.validate_inputs()
        
        print("StatArbX - Statistical Arbitrage Backtesting System")
        print("=" * 60)
        print(f"Tickers: {', '.join(self.args.tickers_list)}")
        print(f"Period: {self.args.start} to {self.args.end}")
        print(f"Initial Capital: ${self.args.capital:,.2f}")
        print("=" * 60)
        
        # Execute pipeline
        try:
            # 1. Download data
            data = self.download_data()
            
            # 2. Find cointegrated pairs
            cointegrated_pairs = self.find_cointegrated_pairs(data)
            
            if not cointegrated_pairs:
                print("\nNo cointegrated pairs found. Try different tickers or adjust parameters.")
                return
            
            # 3. Run backtest
            results, pairs_info, cointegrated_pairs = self.run_backtest(data, cointegrated_pairs)
            
            # 4. Analyze results
            comprehensive_metrics = self.analyze_results(results, pairs_info, cointegrated_pairs)
            
            # 5. Create visualizations
            self.create_visualizations(results, pairs_info, cointegrated_pairs, comprehensive_metrics)
            
            # 6. Save results
            self.save_results(results, pairs_info, cointegrated_pairs, comprehensive_metrics)
            
            print(f"\nBacktest completed successfully!")
            if self.args.save_data or self.args.save_plots:
                print(f"Results saved to: {self.args.output_dir}")
            
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            sys.exit(1)


def main():
    """Entry point for CLI."""
    cli = StatArbCLI()
    cli.run()


if __name__ == "__main__":
    main()
