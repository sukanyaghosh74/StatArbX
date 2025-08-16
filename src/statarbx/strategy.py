"""
Trading strategy module for StatArbX - Z-score based pairs trading strategy.
"""

import pandas as pd
import numpy as np
import backtrader as bt
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsTradingStrategy(bt.Strategy):
    """
    Pairs trading strategy using z-score based entry and exit rules.
    """
    
    params = (
        ('entry_threshold', 2.0),      # Z-score threshold for entry
        ('exit_threshold', 0.0),       # Z-score threshold for exit  
        ('stop_loss', 4.0),           # Stop loss z-score threshold
        ('lookback_period', 20),       # Rolling window for z-score calculation
        ('hedge_ratio', 1.0),         # Hedge ratio between the pair
        ('position_size', 0.1),       # Position size as fraction of portfolio
        ('max_positions', 5),         # Maximum number of concurrent positions
        ('min_trade_distance', 5),    # Minimum days between trades on same pair
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.order = None
        self.positions = {}  # Track positions for each pair
        self.last_trade_date = {}  # Track last trade date for each pair
        self.pair_data = {}  # Store pair-specific data
        
        # Assume we have two data feeds per pair (stock1, stock2)
        # Data feeds should be added in pairs: [stock1_pair1, stock2_pair1, stock1_pair2, stock2_pair2, ...]
        self.pairs = []
        
        # Group data feeds into pairs
        for i in range(0, len(self.datas), 2):
            if i + 1 < len(self.datas):
                pair_name = f"pair_{i//2}"
                data1 = self.datas[i]
                data2 = self.datas[i + 1]
                
                self.pairs.append({
                    'name': pair_name,
                    'data1': data1,
                    'data2': data2,
                    'hedge_ratio': self.params.hedge_ratio,
                })
                
                # Initialize pair-specific tracking
                self.positions[pair_name] = {
                    'position': 0,  # 0: no position, 1: long spread, -1: short spread
                    'entry_price1': 0,
                    'entry_price2': 0,
                    'entry_date': None,
                    'spread_entry': 0,
                    'z_score_entry': 0
                }
                
                self.last_trade_date[pair_name] = None
        
        logger.info(f"Initialized strategy with {len(self.pairs)} pairs")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def calculate_spread_and_zscore(self, pair: Dict, lookback: int) -> Tuple[float, float]:
        """
        Calculate current spread and z-score for a pair.
        
        Args:
            pair: Pair dictionary with data feeds
            lookback: Lookback period for z-score calculation
            
        Returns:
            Tuple of (spread, z_score)
        """
        data1 = pair['data1']
        data2 = pair['data2']
        hedge_ratio = pair['hedge_ratio']
        
        # Get current prices
        price1 = data1.close[0]
        price2 = data2.close[0]
        
        # Calculate current spread
        spread = price1 - hedge_ratio * price2
        
        # Calculate historical spreads for z-score
        spreads = []
        for i in range(min(lookback, len(data1))):
            hist_price1 = data1.close[-i]
            hist_price2 = data2.close[-i]
            hist_spread = hist_price1 - hedge_ratio * hist_price2
            spreads.append(hist_spread)
        
        if len(spreads) < 2:
            return spread, 0.0
        
        # Calculate z-score
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread == 0:
            z_score = 0.0
        else:
            z_score = (spread - mean_spread) / std_spread
        
        return spread, z_score
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.broker.getvalue()
    
    def calculate_position_size(self, price1: float, price2: float, hedge_ratio: float) -> Tuple[int, int]:
        """
        Calculate position sizes for the pair based on available capital.
        
        Args:
            price1: Price of first stock
            price2: Price of second stock
            hedge_ratio: Hedge ratio
            
        Returns:
            Tuple of (size1, size2) - number of shares for each stock
        """
        portfolio_value = self.get_portfolio_value()
        position_value = portfolio_value * self.params.position_size
        
        # Calculate total cost for one unit of the spread
        unit_cost = price1 + hedge_ratio * price2
        
        if unit_cost == 0:
            return 0, 0
        
        # Calculate number of units we can afford
        units = int(position_value / unit_cost)
        
        # Return sizes (negative for short positions)
        size1 = units
        size2 = int(units * hedge_ratio)
        
        return size1, size2
    
    def check_trade_distance(self, pair_name: str) -> bool:
        """
        Check if enough time has passed since last trade.
        
        Args:
            pair_name: Name of the pair
            
        Returns:
            True if enough time has passed
        """
        if self.last_trade_date[pair_name] is None:
            return True
        
        current_date = self.datas[0].datetime.date(0)
        days_since_trade = (current_date - self.last_trade_date[pair_name]).days
        
        return days_since_trade >= self.params.min_trade_distance
    
    def count_active_positions(self) -> int:
        """Count number of active positions."""
        return sum(1 for pos in self.positions.values() if pos['position'] != 0)
    
    def enter_long_spread(self, pair: Dict, spread: float, z_score: float):
        """
        Enter long spread position (buy stock1, sell stock2).
        
        Args:
            pair: Pair dictionary
            spread: Current spread value
            z_score: Current z-score
        """
        pair_name = pair['name']
        data1, data2 = pair['data1'], pair['data2']
        hedge_ratio = pair['hedge_ratio']
        
        price1 = data1.close[0]
        price2 = data2.close[0]
        
        size1, size2 = self.calculate_position_size(price1, price2, hedge_ratio)
        
        if size1 > 0 and size2 > 0:
            # Buy stock1, sell stock2
            self.buy(data=data1, size=size1)
            self.sell(data=data2, size=size2)
            
            # Update position tracking
            self.positions[pair_name].update({
                'position': 1,
                'entry_price1': price1,
                'entry_price2': price2,
                'entry_date': self.datas[0].datetime.date(0),
                'spread_entry': spread,
                'z_score_entry': z_score
            })
            
            self.last_trade_date[pair_name] = self.datas[0].datetime.date(0)
            
            self.log(f'LONG SPREAD {pair_name}: Buy {size1} {data1._name} at {price1:.2f}, '
                    f'Sell {size2} {data2._name} at {price2:.2f}, Z-score: {z_score:.2f}')
    
    def enter_short_spread(self, pair: Dict, spread: float, z_score: float):
        """
        Enter short spread position (sell stock1, buy stock2).
        
        Args:
            pair: Pair dictionary
            spread: Current spread value
            z_score: Current z-score
        """
        pair_name = pair['name']
        data1, data2 = pair['data1'], pair['data2']
        hedge_ratio = pair['hedge_ratio']
        
        price1 = data1.close[0]
        price2 = data2.close[0]
        
        size1, size2 = self.calculate_position_size(price1, price2, hedge_ratio)
        
        if size1 > 0 and size2 > 0:
            # Sell stock1, buy stock2
            self.sell(data=data1, size=size1)
            self.buy(data=data2, size=size2)
            
            # Update position tracking
            self.positions[pair_name].update({
                'position': -1,
                'entry_price1': price1,
                'entry_price2': price2,
                'entry_date': self.datas[0].datetime.date(0),
                'spread_entry': spread,
                'z_score_entry': z_score
            })
            
            self.last_trade_date[pair_name] = self.datas[0].datetime.date(0)
            
            self.log(f'SHORT SPREAD {pair_name}: Sell {size1} {data1._name} at {price1:.2f}, '
                    f'Buy {size2} {data2._name} at {price2:.2f}, Z-score: {z_score:.2f}')
    
    def exit_position(self, pair: Dict, z_score: float, reason: str = "Exit signal"):
        """
        Exit current position for a pair.
        
        Args:
            pair: Pair dictionary
            z_score: Current z-score
            reason: Reason for exit
        """
        pair_name = pair['name']
        data1, data2 = pair['data1'], pair['data2']
        
        position_info = self.positions[pair_name]
        
        if position_info['position'] == 0:
            return  # No position to exit
        
        # Get current position sizes
        pos1 = self.getposition(data1).size
        pos2 = self.getposition(data2).size
        
        # Close positions
        if pos1 != 0:
            self.close(data=data1)
        if pos2 != 0:
            self.close(data=data2)
        
        # Calculate P&L
        current_price1 = data1.close[0]
        current_price2 = data2.close[0]
        entry_price1 = position_info['entry_price1']
        entry_price2 = position_info['entry_price2']
        
        if position_info['position'] == 1:  # Long spread
            pnl = pos1 * (current_price1 - entry_price1) + pos2 * (entry_price2 - current_price2)
        else:  # Short spread
            pnl = pos1 * (entry_price1 - current_price1) + pos2 * (current_price2 - entry_price2)
        
        # Reset position tracking
        self.positions[pair_name].update({
            'position': 0,
            'entry_price1': 0,
            'entry_price2': 0,
            'entry_date': None,
            'spread_entry': 0,
            'z_score_entry': 0
        })
        
        self.log(f'EXIT {pair_name}: {reason}, Z-score: {z_score:.2f}, Est. P&L: {pnl:.2f}')
    
    def next(self):
        """Main strategy logic called for each bar."""
        # Process each pair
        for pair in self.pairs:
            pair_name = pair['name']
            
            # Calculate current spread and z-score
            spread, z_score = self.calculate_spread_and_zscore(pair, self.params.lookback_period)
            
            # Skip if z-score calculation failed
            if np.isnan(z_score) or np.isinf(z_score):
                continue
            
            current_position = self.positions[pair_name]['position']
            
            # Check for exit conditions first
            if current_position != 0:
                # Exit on mean reversion
                if abs(z_score) <= self.params.exit_threshold:
                    self.exit_position(pair, z_score, "Mean reversion")
                
                # Exit on stop loss
                elif abs(z_score) >= self.params.stop_loss:
                    self.exit_position(pair, z_score, "Stop loss")
                
                # Exit if z-score changed direction significantly
                elif (current_position == 1 and z_score > self.params.entry_threshold) or \
                     (current_position == -1 and z_score < -self.params.entry_threshold):
                    self.exit_position(pair, z_score, "Direction change")
            
            # Check for entry conditions
            else:
                # Check constraints
                if (self.count_active_positions() >= self.params.max_positions or
                    not self.check_trade_distance(pair_name)):
                    continue
                
                # Long spread entry (z-score < -entry_threshold)
                if z_score <= -self.params.entry_threshold:
                    self.enter_long_spread(pair, spread, z_score)
                
                # Short spread entry (z-score > entry_threshold)
                elif z_score >= self.params.entry_threshold:
                    self.enter_short_spread(pair, spread, z_score)
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')


class PairsTradingData:
    """Helper class to prepare data for pairs trading strategy."""
    
    @staticmethod
    def create_data_feeds(price_data: pd.DataFrame, pair_info: Dict) -> List[bt.feeds.PandasData]:
        """
        Create Backtrader data feeds for a pair.
        
        Args:
            price_data: DataFrame with stock prices
            pair_info: Dictionary with pair information including tickers and hedge ratio
            
        Returns:
            List of data feeds [data1, data2]
        """
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        
        # Create individual price series
        data1_df = pd.DataFrame(index=price_data.index)
        data1_df['close'] = price_data[ticker1]
        data1_df['open'] = price_data[ticker1]  # Using close as proxy for open
        data1_df['high'] = price_data[ticker1]
        data1_df['low'] = price_data[ticker1]
        data1_df['volume'] = 1000000  # Dummy volume
        
        data2_df = pd.DataFrame(index=price_data.index)
        data2_df['close'] = price_data[ticker2]
        data2_df['open'] = price_data[ticker2]
        data2_df['high'] = price_data[ticker2]
        data2_df['low'] = price_data[ticker2]
        data2_df['volume'] = 1000000  # Dummy volume
        
        # Create Backtrader data feeds
        data1 = bt.feeds.PandasData(
            dataname=data1_df,
            name=ticker1,
            plot=False
        )
        
        data2 = bt.feeds.PandasData(
            dataname=data2_df,
            name=ticker2,
            plot=False
        )
        
        return [data1, data2]
    
    @staticmethod
    def prepare_multiple_pairs(price_data: pd.DataFrame, pairs_info: List[Dict]) -> List[bt.feeds.PandasData]:
        """
        Prepare data feeds for multiple pairs.
        
        Args:
            price_data: DataFrame with stock prices
            pairs_info: List of pair information dictionaries
            
        Returns:
            List of all data feeds
        """
        all_feeds = []
        
        for pair_info in pairs_info:
            feeds = PairsTradingData.create_data_feeds(price_data, pair_info)
            all_feeds.extend(feeds)
        
        return all_feeds


class SignalGenerator:
    """Class for generating trading signals based on z-score analysis."""
    
    @staticmethod
    def generate_signals(spread: pd.Series, entry_threshold: float = 2.0, 
                        exit_threshold: float = 0.0, lookback: int = 20) -> pd.DataFrame:
        """
        Generate trading signals based on z-score.
        
        Args:
            spread: Spread time series
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            lookback: Lookback period for z-score calculation
            
        Returns:
            DataFrame with signals
        """
        # Calculate rolling z-score
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['rolling_mean'] = rolling_mean
        signals['rolling_std'] = rolling_std
        
        # Entry signals
        signals['long_entry'] = z_score <= -entry_threshold
        signals['short_entry'] = z_score >= entry_threshold
        
        # Exit signals
        signals['exit'] = np.abs(z_score) <= exit_threshold
        
        # Stop loss signals (optional)
        signals['stop_loss'] = np.abs(z_score) >= 4.0
        
        return signals.dropna()


if __name__ == "__main__":
    # Example usage
    from .data import DataDownloader
    from .cointegration import CointegrationAnalyzer
    
    # Download sample data
    tickers = ["AAPL", "MSFT"]
    downloader = DataDownloader(tickers, "2020-01-01", "2023-12-31")
    data = downloader.download_data()
    
    # Find cointegrated pairs
    analyzer = CointegrationAnalyzer(data)
    pairs = analyzer.find_cointegrated_pairs()
    
    if pairs:
        # Generate signals for first pair
        pair = pairs[0]
        spread = pair['spread']
        
        signal_gen = SignalGenerator()
        signals = signal_gen.generate_signals(spread)
        
        print("Signal Summary:")
        print(f"Long entries: {signals['long_entry'].sum()}")
        print(f"Short entries: {signals['short_entry'].sum()}")
        print(f"Exit signals: {signals['exit'].sum()}")
        print(f"Stop losses: {signals['stop_loss'].sum()}")
    else:
        print("No cointegrated pairs found for signal generation.")
