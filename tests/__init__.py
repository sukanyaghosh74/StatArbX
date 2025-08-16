"""
Test suite for StatArbX - Statistical Arbitrage Trading Bot.
"""

import sys
import os
from pathlib import Path

# Add src directory to path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Test configuration
TEST_DATA_DIR = test_dir / "data"
TEST_OUTPUT_DIR = test_dir / "output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test constants
DEFAULT_TEST_TICKERS = ["AAPL", "MSFT"]
DEFAULT_TEST_START = "2020-01-01"
DEFAULT_TEST_END = "2022-12-31"
DEFAULT_TEST_CAPITAL = 10000
