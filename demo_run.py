#!/usr/bin/env python3
"""
Demo script for StatArbX - Statistical Arbitrage Trading Bot
This script demonstrates a working example with ETFs that are likely to be cointegrated.
"""

import sys
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')

from statarbx import DataDownloader, CointegrationAnalyzer, StatArbBacktester
import pandas as pd

def main():
    print("=" * 60)
    print("StatArbX Demo - Statistical Arbitrage Trading Bot")
    print("=" * 60)
    
    # Use ETFs that track similar indices and are likely cointegrated
    tickers = ['SPY', 'VTI', 'IVV']  # All track S&P 500
    print(f"Testing with tickers: {tickers}")
    print(f"Strategy: Statistical Arbitrage (Pairs Trading)")
    print()
    
    # Download data
    print("📊 Step 1: Downloading market data...")
    downloader = DataDownloader(tickers, '2022-01-01', '2023-12-31')
    data = downloader.download_data()
    print(f"   ✅ Downloaded {data.shape[0]} trading days")
    print(f"   📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    # Analyze correlations
    print("🔍 Step 2: Analyzing correlations...")
    correlation_matrix = data.corr()
    print("   Correlation Matrix:")
    print(correlation_matrix.round(4))
    print()
    
    # Test cointegration with relaxed parameters
    print("📈 Step 3: Testing for cointegration...")
    analyzer = CointegrationAnalyzer(data, significance_level=0.1)
    
    # Test individual pairs manually
    pairs_to_test = [('SPY', 'VTI'), ('SPY', 'IVV'), ('VTI', 'IVV')]
    cointegrated_pairs = []
    
    for ticker1, ticker2 in pairs_to_test:
        print(f"   Testing {ticker1} vs {ticker2}...")
        try:
            result = analyzer.analyze_pair(ticker1, ticker2)
            if 'error' not in result and result['is_cointegrated']:
                print(f"   ✅ {ticker1}-{ticker2}: Cointegrated! (p-value: {result['p_value']:.4f})")
                cointegrated_pairs.append({
                    'name': f"{ticker1}-{ticker2}",
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'hedge_ratio': result['hedge_ratio'],
                    'p_value': result['p_value']
                })
            else:
                print(f"   ❌ {ticker1}-{ticker2}: Not cointegrated (p-value: {result.get('p_value', 'N/A')})")
        except Exception as e:
            print(f"   ⚠️  {ticker1}-{ticker2}: Error - {str(e)}")
    
    if not cointegrated_pairs:
        print("\n⚠️  No cointegrated pairs found for backtesting.")
        print("   This is normal - cointegration relationships are not always present.")
        print("   Try different tickers, time periods, or parameters.")
        return
    
    print(f"\n🎯 Found {len(cointegrated_pairs)} cointegrated pair(s) for trading!")
    
    # Run a simple backtest demonstration
    print("\n🚀 Step 4: Running backtest demonstration...")
    print("   Strategy Parameters:")
    print("   - Entry Threshold: ±2.0 (enter when z-score exceeds this)")
    print("   - Exit Threshold: 0.0 (exit when z-score returns to mean)")
    print("   - Stop Loss: ±4.0 (risk management)")
    print("   - Initial Capital: $100,000")
    print("   - Position Size: 10% per trade")
    
    try:
        backtester = StatArbBacktester(initial_capital=100000)
        
        # Use the first cointegrated pair for backtesting
        pair_info = cointegrated_pairs[0]
        print(f"\n   Trading pair: {pair_info['name']}")
        print(f"   Hedge ratio: {pair_info['hedge_ratio']:.4f}")
        
        strategy_params = {
            'entry_threshold': 2.0,
            'exit_threshold': 0.0,
            'stop_loss': 4.0,
            'position_size': 0.1,
            'lookback_period': 20
        }
        
        # Run backtest
        results = backtester.run_backtest(
            price_data=data,
            pairs_info=[pair_info],
            strategy_params=strategy_params,
            commission=0.001,
            slippage=0.0005
        )
        
        print("\n📊 Backtest Results:")
        print("=" * 40)
        if results:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'pct' in key.lower() or 'ratio' in key.lower():
                        print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
                    elif 'value' in key.lower() or 'capital' in key.lower():
                        print(f"   {key.replace('_', ' ').title()}: ${value:,.2f}")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
    except Exception as e:
        print(f"   ⚠️  Backtesting error: {str(e)}")
        print("   The cointegration analysis worked, but backtesting encountered an issue.")
    
    print("\n✅ StatArbX Demo Complete!")
    print("\nNext Steps:")
    print("- Try different ticker combinations")
    print("- Adjust strategy parameters")
    print("- Use longer time periods for more robust results")
    print("- Add visualization with --save-plots flag")

if __name__ == "__main__":
    main()
