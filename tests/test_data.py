"""
Test suite for the data module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from statarbx.data import DataDownloader, DataPreprocessor


class TestDataDownloader:
    """Test class for DataDownloader."""
    
    def test_init(self):
        """Test DataDownloader initialization."""
        tickers = ["AAPL", "MSFT"]
        start_date = "2020-01-01"
        end_date = "2022-12-31"
        
        downloader = DataDownloader(tickers, start_date, end_date)
        
        assert downloader.tickers == tickers
        assert downloader.start_date == start_date
        assert downloader.end_date == end_date
        assert downloader.data is None
    
    def test_get_trading_days_no_data(self):
        """Test get_trading_days with no data."""
        downloader = DataDownloader(["AAPL"], "2020-01-01", "2022-12-31")
        
        with pytest.raises(ValueError, match="No data available"):
            downloader.get_trading_days()
    
    def test_get_date_range_no_data(self):
        """Test get_date_range with no data."""
        downloader = DataDownloader(["AAPL"], "2020-01-01", "2022-12-31")
        
        with pytest.raises(ValueError, match="No data available"):
            downloader.get_date_range()
    
    @patch('statarbx.data.yf.download')
    def test_download_data_success(self, mock_download):
        """Test successful data download."""
        # Mock yfinance data
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        mock_data = pd.DataFrame({
            ('AAPL', 'Close'): [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            ('MSFT', 'Close'): [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('AAPL', 'Close'), ('MSFT', 'Close')
        ])
        mock_download.return_value = mock_data
        
        downloader = DataDownloader(["AAPL", "MSFT"], "2020-01-01", "2020-01-10")
        result = downloader.download_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert 'AAPL' in result.columns
        assert 'MSFT' in result.columns
        assert len(result) == 10
    
    def test_get_returns_log(self):
        """Test log returns calculation."""
        # Create mock data
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104],
            'MSFT': [200, 202, 204, 206, 208]
        }, index=dates)
        
        downloader = DataDownloader(["AAPL", "MSFT"], "2020-01-01", "2020-01-05")
        downloader.data = data
        
        returns = downloader.get_returns(method='log')
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == 4  # One less than original due to diff
        assert np.allclose(returns['AAPL'].iloc[0], np.log(101/100))
    
    def test_get_returns_simple(self):
        """Test simple returns calculation."""
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104],
            'MSFT': [200, 202, 204, 206, 208]
        }, index=dates)
        
        downloader = DataDownloader(["AAPL", "MSFT"], "2020-01-01", "2020-01-05")
        downloader.data = data
        
        returns = downloader.get_returns(method='simple')
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == 4
        assert np.allclose(returns['AAPL'].iloc[0], 0.01)  # (101-100)/100
    
    def test_get_returns_invalid_method(self):
        """Test get_returns with invalid method."""
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        data = pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]}, index=dates)
        
        downloader = DataDownloader(["AAPL"], "2020-01-01", "2020-01-05")
        downloader.data = data
        
        with pytest.raises(ValueError, match="Method must be"):
            downloader.get_returns(method='invalid')
    
    def test_check_data_quality(self):
        """Test data quality check."""
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        data = pd.DataFrame({
            'AAPL': [100, 101, np.nan, 103, 104],  # Contains NaN
            'MSFT': [200, 0, 202, 203, 204]         # Contains zero
        }, index=dates)
        
        downloader = DataDownloader(["AAPL", "MSFT"], "2020-01-01", "2020-01-05")
        downloader.data = data
        
        quality = downloader.check_data_quality()
        
        assert 'AAPL' in quality
        assert 'MSFT' in quality
        assert quality['AAPL']['missing_values'] == 1
        assert quality['MSFT']['zero_values'] == 1
        assert quality['AAPL']['total_observations'] == 5


class TestDataPreprocessor:
    """Test class for DataPreprocessor."""
    
    def test_normalize_prices_first(self):
        """Test normalize_prices with first method."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        prices = pd.DataFrame({
            'AAPL': [100, 110, 120],
            'MSFT': [200, 220, 240]
        }, index=dates)
        
        normalized = DataPreprocessor.normalize_prices(prices, method='first')
        
        assert np.allclose(normalized.iloc[0], [1.0, 1.0])  # First values should be 1
        assert np.allclose(normalized.iloc[1], [1.1, 1.1])   # 10% increase
        assert np.allclose(normalized.iloc[2], [1.2, 1.2])   # 20% increase
    
    def test_normalize_prices_base100(self):
        """Test normalize_prices with base100 method."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        prices = pd.DataFrame({
            'AAPL': [100, 110, 120],
            'MSFT': [200, 220, 240]
        }, index=dates)
        
        normalized = DataPreprocessor.normalize_prices(prices, method='base100')
        
        assert np.allclose(normalized.iloc[0], [100.0, 100.0])  # First values should be 100
        assert np.allclose(normalized.iloc[1], [110.0, 110.0])   # 10% increase from 100
    
    def test_normalize_prices_invalid_method(self):
        """Test normalize_prices with invalid method."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        prices = pd.DataFrame({'AAPL': [100, 110, 120]}, index=dates)
        
        with pytest.raises(ValueError, match="Method must be"):
            DataPreprocessor.normalize_prices(prices, method='invalid')
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        price1 = pd.Series([100, 110, 120], index=dates, name='AAPL')
        price2 = pd.Series([50, 55, 60], index=dates, name='MSFT')
        hedge_ratio = 2.0
        
        spread = DataPreprocessor.calculate_spread(price1, price2, hedge_ratio)
        
        expected = price1 - hedge_ratio * price2
        assert np.allclose(spread, expected)
        assert spread.iloc[0] == 100 - 2.0 * 50  # 0
    
    def test_calculate_zscore(self):
        """Test z-score calculation."""
        dates = pd.date_range('2020-01-01', '2020-01-30', freq='D')
        # Create a series with known mean and std
        spread = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        
        zscore = DataPreprocessor.calculate_zscore(spread, window=20)
        
        # Z-score should be calculated for periods after window
        assert len(zscore.dropna()) <= len(spread) - 20 + 1
        # First few values should be NaN due to rolling window
        assert zscore.iloc[:19].isna().all()
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        # Create data with outliers
        data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104, 105, 106, 107, 108, 1000]  # Last value is outlier
        }, index=dates)
        
        cleaned = DataPreprocessor.remove_outliers(data, method='iqr')
        
        # Should have fewer rows due to outlier removal
        assert len(cleaned) < len(data)
    
    def test_remove_outliers_zscore(self):
        """Test outlier removal using z-score method."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        # Create data with outliers
        data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104, 105, 106, 107, 108, 1000]  # Last value is outlier
        }, index=dates)
        
        cleaned = DataPreprocessor.remove_outliers(data, method='zscore', threshold=2.0)
        
        # Should have fewer rows due to outlier removal
        assert len(cleaned) < len(data)
    
    def test_remove_outliers_invalid_method(self):
        """Test outlier removal with invalid method."""
        dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        data = pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]}, index=dates)
        
        with pytest.raises(ValueError, match="Method must be"):
            DataPreprocessor.remove_outliers(data, method='invalid')


@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)
    
    # Generate correlated price series
    returns1 = np.random.normal(0.0005, 0.02, len(dates))
    returns2 = 0.8 * returns1 + 0.6 * np.random.normal(0.0005, 0.02, len(dates))
    
    prices1 = 100 * np.cumprod(1 + returns1)
    prices2 = 200 * np.cumprod(1 + returns2)
    
    return pd.DataFrame({
        'AAPL': prices1,
        'MSFT': prices2
    }, index=dates)


def test_data_downloader_integration(sample_data):
    """Integration test for DataDownloader with sample data."""
    # Mock the download to return our sample data
    downloader = DataDownloader(["AAPL", "MSFT"], "2020-01-01", "2020-12-31")
    downloader.data = sample_data
    
    # Test various methods
    trading_days = downloader.get_trading_days()
    date_range = downloader.get_date_range()
    returns = downloader.get_returns()
    quality = downloader.check_data_quality()
    
    assert trading_days == len(sample_data)
    assert isinstance(date_range, tuple)
    assert len(date_range) == 2
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(quality, dict)
    assert len(quality) == 2
