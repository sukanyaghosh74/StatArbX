#!/usr/bin/env python3
"""
Simple test script for StatArbX
"""

import sys
sys.path.insert(0, 'src')

from statarbx import DataDownloader, CointegrationAnalyzer
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Testing StatArbX System...")
    
    # Test with two highly correlated ETFs
    tickers = ['SPY', 'VTI']  # Both track S&P 500
    print(f"Testing with tickers: {tickers}")
    
    # Download data
    downloader = DataDownloader(tickers, '2022-01-01', '2023-12-31')
    data = downloader.download_data()
    print(f"Downloaded data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Check correlation
    correlation = data.corr()
    print(f"Correlation matrix:\n{correlation}")
    
    # Test cointegration analysis
    analyzer = CointegrationAnalyzer(data, significance_level=0.1)
    
    # Manual pair analysis
    print("\nManual pair analysis:")
    try:
        result = analyzer.analyze_pair('SPY', 'VTI')
        if 'error' not in result:
            print(f"P-value: {result['p_value']:.4f}")
            print(f"Hedge ratio: {result['hedge_ratio']:.4f}")
            print(f"Is cointegrated: {result['is_cointegrated']}")
            print(f"Half-life: {result['half_life']:.2f}")
        else:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
    
    # Find all pairs
    print("\nFinding all cointegrated pairs:")
    pairs = analyzer.find_cointegrated_pairs(min_half_life=0.1, max_half_life=1000)
    print(f"Found {len(pairs)} cointegrated pairs")
    
    if pairs:
        pair = pairs[0]
        print(f"Best pair: {pair['ticker1']}-{pair['ticker2']}")
        print(f"P-value: {pair['p_value']:.4f}")
        print(f"Half-life: {pair['half_life']:.2f}")

if __name__ == "__main__":
    main()
