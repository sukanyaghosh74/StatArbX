"""
Cointegration analysis module for StatArbX - Engle-Granger test and pairs analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import itertools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CointegrationAnalyzer:
    """Class for performing cointegration analysis on stock pairs."""
    
    def __init__(self, price_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize CointegrationAnalyzer.
        
        Args:
            price_data: DataFrame with stock price data
            significance_level: Significance level for cointegration tests
        """
        self.price_data = price_data
        self.significance_level = significance_level
        self.pairs_results = {}
        
    def engle_granger_test(self, y: pd.Series, x: pd.Series) -> Dict[str, any]:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            
        Returns:
            Dictionary with test results
        """
        # Step 1: Run cointegration regression
        x_with_const = sm.add_constant(x)
        model = OLS(y, x_with_const).fit()
        
        # Extract coefficients
        alpha = model.params[0]  # intercept
        beta = model.params[1]   # hedge ratio
        
        # Step 2: Test residuals for stationarity
        residuals = model.resid
        adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals, autolag='AIC')
        
        # Determine if cointegrated
        is_cointegrated = p_value < self.significance_level
        
        return {
            'alpha': alpha,
            'beta': beta,
            'hedge_ratio': beta,
            'residuals': residuals,
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_cointegrated': is_cointegrated,
            'r_squared': model.rsquared,
            'regression_model': model
        }
    
    def statsmodels_coint_test(self, y: pd.Series, x: pd.Series) -> Dict[str, any]:
        """
        Use statsmodels cointegration test as alternative verification.
        
        Args:
            y: First time series
            x: Second time series
            
        Returns:
            Dictionary with test results
        """
        try:
            coint_stat, p_value, critical_values = coint(y, x)
            is_cointegrated = p_value < self.significance_level
            
            # Calculate hedge ratio using OLS
            x_with_const = sm.add_constant(x)
            model = OLS(y, x_with_const).fit()
            hedge_ratio = model.params[1]
            
            return {
                'coint_statistic': coint_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_cointegrated': is_cointegrated,
                'hedge_ratio': hedge_ratio
            }
        except Exception as e:
            logger.error(f"Error in statsmodels cointegration test: {str(e)}")
            return {'error': str(e)}
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for a spread.
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in trading days
        """
        # Create lagged spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Ensure same length
        min_len = min(len(spread_lag), len(spread_diff))
        spread_lag = spread_lag.iloc[-min_len:]
        spread_diff = spread_diff.iloc[-min_len:]
        
        # Run regression: Δspread_t = α + β * spread_{t-1} + ε_t
        try:
            X = sm.add_constant(spread_lag)
            model = OLS(spread_diff, X).fit()
            beta = model.params[1]
            
            if beta >= 0:
                # No mean reversion
                return np.inf
            
            # Half-life = ln(0.5) / ln(1 + β)
            half_life = -np.log(2) / np.log(1 + beta)
            return max(half_life, 0)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating half-life: {str(e)}")
            return np.inf
    
    def analyze_pair(self, ticker1: str, ticker2: str) -> Dict[str, any]:
        """
        Perform comprehensive analysis of a stock pair.
        
        Args:
            ticker1: First stock ticker
            ticker2: Second stock ticker
            
        Returns:
            Dictionary with pair analysis results
        """
        if ticker1 not in self.price_data.columns or ticker2 not in self.price_data.columns:
            raise ValueError(f"Tickers {ticker1} and/or {ticker2} not found in price data")
        
        price1 = self.price_data[ticker1].dropna()
        price2 = self.price_data[ticker2].dropna()
        
        # Ensure same length
        common_dates = price1.index.intersection(price2.index)
        price1 = price1.loc[common_dates]
        price2 = price2.loc[common_dates]
        
        if len(price1) < 30:
            logger.warning(f"Insufficient data for pair {ticker1}-{ticker2}: {len(price1)} observations")
            return {'error': 'Insufficient data'}
        
        # Test both directions
        results_1_2 = self.engle_granger_test(price1, price2)
        results_2_1 = self.engle_granger_test(price2, price1)
        
        # Use the direction with better cointegration
        if results_1_2['p_value'] <= results_2_1['p_value']:
            primary_result = results_1_2
            y_ticker, x_ticker = ticker1, ticker2
        else:
            primary_result = results_2_1
            y_ticker, x_ticker = ticker2, ticker1
            # Invert hedge ratio for consistency
            primary_result['hedge_ratio'] = 1 / primary_result['hedge_ratio']
        
        # Calculate spread and half-life
        hedge_ratio = primary_result['hedge_ratio']
        if y_ticker == ticker1:
            spread = price1 - hedge_ratio * price2
        else:
            spread = price2 - hedge_ratio * price1
            
        half_life = self.calculate_half_life(spread)
        
        # Additional statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        spread_min = spread.min()
        spread_max = spread.max()
        
        # Correlation
        correlation = price1.corr(price2)
        
        # Verify with statsmodels
        sm_result = self.statsmodels_coint_test(price1, price2)
        
        result = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'y_ticker': y_ticker,
            'x_ticker': x_ticker,
            'hedge_ratio': hedge_ratio,
            'is_cointegrated': primary_result['is_cointegrated'],
            'p_value': primary_result['p_value'],
            'adf_statistic': primary_result['adf_statistic'],
            'critical_values': primary_result['critical_values'],
            'r_squared': primary_result['r_squared'],
            'half_life': half_life,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_min': spread_min,
            'spread_max': spread_max,
            'correlation': correlation,
            'observations': len(price1),
            'statsmodels_verification': sm_result
        }
        
        return result
    
    def find_cointegrated_pairs(self, min_half_life: float = 1, max_half_life: float = 252) -> List[Dict[str, any]]:
        """
        Find all cointegrated pairs in the dataset.
        
        Args:
            min_half_life: Minimum acceptable half-life (trading days)
            max_half_life: Maximum acceptable half-life (trading days)
            
        Returns:
            List of cointegrated pairs with their analysis results
        """
        tickers = list(self.price_data.columns)
        cointegrated_pairs = []
        
        logger.info(f"Testing {len(tickers)} tickers for cointegration...")
        logger.info(f"Total pairs to test: {len(list(itertools.combinations(tickers, 2)))}")
        
        for i, (ticker1, ticker2) in enumerate(itertools.combinations(tickers, 2)):
            try:
                result = self.analyze_pair(ticker1, ticker2)
                
                if 'error' not in result and result['is_cointegrated']:
                    # Filter by half-life
                    if min_half_life <= result['half_life'] <= max_half_life:
                        cointegrated_pairs.append(result)
                        logger.info(f"Found cointegrated pair: {ticker1}-{ticker2} "
                                  f"(p-value: {result['p_value']:.4f}, "
                                  f"half-life: {result['half_life']:.2f})")
                
                # Store all results for later analysis
                self.pairs_results[f"{ticker1}_{ticker2}"] = result
                
            except Exception as e:
                logger.error(f"Error analyzing pair {ticker1}-{ticker2}: {str(e)}")
                continue
                
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} pairs...")
        
        # Sort by p-value (best first)
        cointegrated_pairs.sort(key=lambda x: x['p_value'])
        
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        return cointegrated_pairs
    
    def get_pair_summary(self, cointegrated_pairs: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Create summary DataFrame of cointegrated pairs.
        
        Args:
            cointegrated_pairs: List of cointegrated pair results
            
        Returns:
            Summary DataFrame
        """
        if not cointegrated_pairs:
            return pd.DataFrame()
            
        summary_data = []
        for pair in cointegrated_pairs:
            summary_data.append({
                'Pair': f"{pair['ticker1']}-{pair['ticker2']}",
                'Y_Ticker': pair['y_ticker'],
                'X_Ticker': pair['x_ticker'],
                'Hedge_Ratio': round(pair['hedge_ratio'], 4),
                'P_Value': round(pair['p_value'], 4),
                'ADF_Statistic': round(pair['adf_statistic'], 4),
                'R_Squared': round(pair['r_squared'], 4),
                'Half_Life': round(pair['half_life'], 2),
                'Correlation': round(pair['correlation'], 4),
                'Spread_Mean': round(pair['spread_mean'], 4),
                'Spread_Std': round(pair['spread_std'], 4),
                'Observations': pair['observations']
            })
        
        return pd.DataFrame(summary_data)


class SpreadAnalyzer:
    """Class for analyzing spread characteristics of cointegrated pairs."""
    
    @staticmethod
    def calculate_spread_statistics(spread: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive spread statistics.
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with spread statistics
        """
        return {
            'mean': spread.mean(),
            'std': spread.std(),
            'min': spread.min(),
            'max': spread.max(),
            'median': spread.median(),
            'skewness': spread.skew(),
            'kurtosis': spread.kurtosis(),
            'q25': spread.quantile(0.25),
            'q75': spread.quantile(0.75),
            'range': spread.max() - spread.min(),
            'iqr': spread.quantile(0.75) - spread.quantile(0.25)
        }
    
    @staticmethod
    def identify_trading_opportunities(spread: pd.Series, z_score: pd.Series, 
                                     entry_threshold: float = 2.0, 
                                     exit_threshold: float = 0.0) -> pd.DataFrame:
        """
        Identify potential trading opportunities based on z-score.
        
        Args:
            spread: Spread time series
            z_score: Z-score time series
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        
        # Generate signals
        signals['long_entry'] = z_score < -entry_threshold
        signals['short_entry'] = z_score > entry_threshold
        signals['exit_signal'] = np.abs(z_score) < exit_threshold
        
        return signals


if __name__ == "__main__":
    # Example usage
    from .data import DataDownloader
    
    # Download sample data
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    downloader = DataDownloader(tickers, "2020-01-01", "2023-12-31")
    data = downloader.download_data()
    
    # Analyze cointegration
    analyzer = CointegrationAnalyzer(data)
    cointegrated_pairs = analyzer.find_cointegrated_pairs()
    
    # Display results
    if cointegrated_pairs:
        summary = analyzer.get_pair_summary(cointegrated_pairs)
        print("Cointegrated Pairs Summary:")
        print(summary.to_string(index=False))
    else:
        print("No cointegrated pairs found.")
