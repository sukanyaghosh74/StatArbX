"""
Configuration module for StatArbX - Default settings and configuration management.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data downloading and processing."""
    default_start_date: str = "2018-01-01"
    default_end_date: str = "2024-12-31"
    default_tickers: List[str] = None
    data_cache_dir: str = "cache"
    max_cache_age_days: int = 7
    
    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]


@dataclass 
class CointegrationConfig:
    """Configuration for cointegration analysis."""
    significance_level: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 252.0
    min_observations: int = 30
    test_both_directions: bool = True
    use_adf_critical_values: bool = True


@dataclass
class StrategyConfig:
    """Configuration for trading strategy parameters."""
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    stop_loss_threshold: float = 4.0
    lookback_period: int = 20
    position_size: float = 0.1
    max_positions: int = 5
    min_trade_distance: int = 5
    
    # Risk management
    max_position_value: float = 0.2
    max_daily_loss: float = 0.05
    min_cash_reserve: float = 0.1
    volatility_adjustment: bool = True
    correlation_threshold: float = 0.7


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    margin: Optional[float] = None
    interest_rate: float = 0.02  # 2% annual
    
    # Performance tracking
    benchmark_ticker: str = "SPY"
    track_portfolio_values: bool = True
    track_trades: bool = True
    track_positions: bool = True


@dataclass
class OutputConfig:
    """Configuration for output and visualization."""
    output_dir: str = "out"
    save_plots: bool = False
    show_plots: bool = False
    save_data: bool = False
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # Report settings
    generate_html_report: bool = False
    include_pair_analysis: bool = True
    max_pairs_in_report: int = 5


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    suppress_warnings: bool = False


@dataclass
class StatArbConfig:
    """Main configuration class containing all sub-configurations."""
    data: DataConfig = None
    cointegration: CointegrationConfig = None
    strategy: StrategyConfig = None
    backtest: BacktestConfig = None
    output: OutputConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.cointegration is None:
            self.cointegration = CointegrationConfig()
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StatArbConfig':
        """Create configuration from dictionary."""
        # Extract sub-configs
        data_config = DataConfig(**config_dict.get('data', {}))
        cointegration_config = CointegrationConfig(**config_dict.get('cointegration', {}))
        strategy_config = StrategyConfig(**config_dict.get('strategy', {}))
        backtest_config = BacktestConfig(**config_dict.get('backtest', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            data=data_config,
            cointegration=cointegration_config,
            strategy=strategy_config,
            backtest=backtest_config,
            output=output_config,
            logging=logging_config
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'StatArbConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (DataConfig, CointegrationConfig, 
                                                 StrategyConfig, BacktestConfig, 
                                                 OutputConfig, LoggingConfig)):
                    # Update sub-config
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(self, key, value)


class ConfigManager:
    """Configuration manager for StatArbX."""
    
    DEFAULT_CONFIG_FILE = "config.json"
    ENV_PREFIX = "STATARBX_"
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self._config = None
    
    def load_config(self) -> StatArbConfig:
        """Load configuration from file and environment variables."""
        # Start with default configuration
        config = StatArbConfig()
        
        # Load from file if it exists
        if os.path.exists(self.config_file):
            config = StatArbConfig.from_json(self.config_file)
        
        # Override with environment variables
        self._load_from_env(config)
        
        self._config = config
        return config
    
    def _load_from_env(self, config: StatArbConfig):
        """Load configuration from environment variables."""
        env_mappings = {
            f"{self.ENV_PREFIX}INITIAL_CAPITAL": ("backtest", "initial_capital", float),
            f"{self.ENV_PREFIX}COMMISSION": ("backtest", "commission", float),
            f"{self.ENV_PREFIX}SLIPPAGE": ("backtest", "slippage", float),
            f"{self.ENV_PREFIX}ENTRY_THRESHOLD": ("strategy", "entry_threshold", float),
            f"{self.ENV_PREFIX}EXIT_THRESHOLD": ("strategy", "exit_threshold", float),
            f"{self.ENV_PREFIX}POSITION_SIZE": ("strategy", "position_size", float),
            f"{self.ENV_PREFIX}OUTPUT_DIR": ("output", "output_dir", str),
            f"{self.ENV_PREFIX}LOG_LEVEL": ("logging", "level", str),
            f"{self.ENV_PREFIX}SAVE_PLOTS": ("output", "save_plots", self._str_to_bool),
            f"{self.ENV_PREFIX}SAVE_DATA": ("output", "save_data", self._str_to_bool),
        }
        
        for env_var, (section, param, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    setattr(getattr(config, section), param, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {env_var}={value}: {e}")
    
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def save_config(self, config: StatArbConfig = None):
        """Save configuration to file."""
        config = config or self._config
        if config:
            config.to_json(self.config_file)
    
    def get_config(self) -> StatArbConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config


# Global configuration instance
_config_manager = ConfigManager()

def get_config() -> StatArbConfig:
    """Get the global configuration instance."""
    return _config_manager.get_config()

def load_config(config_file: str = None) -> StatArbConfig:
    """Load configuration from specified file."""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager.load_config()

def save_config(config: StatArbConfig, config_file: str = None):
    """Save configuration to specified file."""
    config_file = config_file or ConfigManager.DEFAULT_CONFIG_FILE
    config.to_json(config_file)


# Example configuration presets
CONSERVATIVE_PRESET = {
    "strategy": {
        "entry_threshold": 2.5,
        "exit_threshold": 0.5,
        "stop_loss_threshold": 3.5,
        "position_size": 0.05,
        "max_positions": 3
    },
    "backtest": {
        "commission": 0.002,
        "slippage": 0.001
    }
}

AGGRESSIVE_PRESET = {
    "strategy": {
        "entry_threshold": 1.5,
        "exit_threshold": 0.0,
        "stop_loss_threshold": 4.0,
        "position_size": 0.15,
        "max_positions": 8
    },
    "backtest": {
        "commission": 0.0005,
        "slippage": 0.0003
    }
}

PRESETS = {
    "conservative": CONSERVATIVE_PRESET,
    "aggressive": AGGRESSIVE_PRESET,
    "default": {}
}

def apply_preset(preset_name: str) -> StatArbConfig:
    """Apply a configuration preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    config = StatArbConfig()
    preset_dict = PRESETS[preset_name]
    
    if preset_dict:
        config.update(**preset_dict)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = StatArbConfig()
    
    print("Default Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save to file
    config.to_json("example_config.json")
    
    # Load from file
    loaded_config = StatArbConfig.from_json("example_config.json")
    
    # Apply preset
    conservative_config = apply_preset("conservative")
    print("\nConservative Preset:")
    print(json.dumps(conservative_config.to_dict(), indent=2))
