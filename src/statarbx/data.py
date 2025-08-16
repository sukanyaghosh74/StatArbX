"""
Data module for StatArbX - Stock data downloading and preprocessing.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Class for downloading and preprocessing stock data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize DataDownloader.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Returns:
            DataFrame with stock prices
        """
        logger.info(f"Downloading data for tickers: {self.tickers}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        try:
            # Download data for all tickers
            data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if len(self.tickers) == 1:
                # If only one ticker, yfinance returns different structure
                data.columns = pd.MultiIndex.from_product([self.tickers, data.columns])
            
            # Extract adjusted close prices
            close_prices = {}
            for ticker in self.tickers:
                try:
                    close_prices[ticker] = data[ticker]['Close']
                except KeyError:
                    logger.error(f"Could not find Close price for {ticker}")
                    continue
            
            self.data = pd.DataFrame(close_prices)
            self.data = self.data.dropna()
            
            logger.info(f"Downloaded {len(self.data)} trading days of data")
            logger.info(f"Data shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
            
    def get_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
            
        if method == 'log':
            returns = np.log(self.data / self.data.shift(1))
        elif method == 'simple':
            returns = self.data.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'simple'")
            
        return returns.dropna()
    
    def check_data_quality(self) -> Dict[str, any]:
        """
        Check data quality and return statistics.
        
        Returns:
            Dictionary with data quality metrics
        """
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
            
        quality_metrics = {}
        
        for ticker in self.data.columns:
            ticker_data = self.data[ticker]
            
            quality_metrics[ticker] = {
                'total_observations': len(ticker_data),
                'missing_values': ticker_data.isnull().sum(),
                'zero_values': (ticker_data == 0).sum(),
                'negative_values': (ticker_data < 0).sum(),
                'start_price': ticker_data.iloc[0],
                'end_price': ticker_data.iloc[-1],
                'min_price': ticker_data.min(),
                'max_price': ticker_data.max(),
                'mean_price': ticker_data.mean(),
                'std_price': ticker_data.std()
            }
            
        return quality_metrics
    
    def get_trading_days(self) -> int:
        """Get number of trading days in the dataset."""
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
        return len(self.data)
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get actual date range of the data."""
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
        return self.data.index[0], self.data.index[-1]


class DataPreprocessor:
    """Class for preprocessing stock data for pairs trading."""
    
    @staticmethod
    def normalize_prices(prices: pd.DataFrame, method: str = 'first') -> pd.DataFrame:
        """
        Normalize prices to start at 100 or first value.
        
        Args:
            prices: DataFrame with price data
            method: 'first' to normalize to first value, 'base100' to start at 100
            
        Returns:
            Normalized price DataFrame
        """
        if method == 'first':
            return prices / prices.iloc[0]
        elif method == 'base100':
            return (prices / prices.iloc[0]) * 100
        else:
            raise ValueError("Method must be 'first' or 'base100'")
    
    @staticmethod
    def calculate_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: float) -> pd.Series:
        """
        Calculate the spread between two price series.
        
        Args:
            price1: First price series
            price2: Second price series
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Spread series
        """
        return price1 - hedge_ratio * price2
    
    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling z-score of a series.
        
        Args:
            series: Input series
            window: Rolling window size
            
        Returns:
            Z-score series
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            data: Input DataFrame
            method: 'iqr' for interquartile range, 'zscore' for z-score method
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        cleaned_data = data.copy()
        
        for column in data.columns:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                mask = z_scores <= threshold
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")
                
            cleaned_data.loc[~mask, column] = np.nan
            
        return cleaned_data.dropna()


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT"]
    downloader = DataDownloader(tickers, "2020-01-01", "2023-12-31")
    data = downloader.download_data()
    
    quality = downloader.check_data_quality()
    print("Data Quality Metrics:")
    for ticker, metrics in quality.items():
        print(f"\n{ticker}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
