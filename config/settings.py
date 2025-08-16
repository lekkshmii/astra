"""Global Configuration - Astra Trading Platform.
=============================================

Centralized configuration management.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    type: str = "sqlite"
    path: str = "data/astra.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None

@dataclass
class DataConfig:
    """Data source configuration."""

    primary_source: str = "yfinance"
    cache_enabled: bool = True
    cache_duration_hours: int = 24
    max_retries: int = 3
    timeout_seconds: int = 30

@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    benchmark: str = "SPY"
    frequency: str = "D"

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""

    default_simulations: int = 10000
    default_time_steps: int = 252
    random_seed: Optional[int] = 42
    use_rust_engine: bool = True
    parallel_workers: int = 4

@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_drawdown: float = 0.15
    max_daily_loss: float = 0.05
    max_position_weight: float = 0.30
    volatility_threshold: float = 3.0
    correlation_threshold: float = 0.8

@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    style: str = "professional"
    dpi: int = 300
    figure_size: tuple = (12, 8)
    color_scheme: str = "astra"
    save_format: str = "png"

class Settings:
    """Global settings manager for Astra Trading Platform.

    Loads configuration from environment variables and config files.
    """

    def __init__(self, config_file: Optional[str] = None) -> None:
        """Initialize settings."""
        self.config_file = config_file
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from various sources."""
        # Default configuration
        self.database = DatabaseConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.monte_carlo = MonteCarloConfig()
        self.risk = RiskConfig()
        self.visualization = VisualizationConfig()

        # Load from config file if provided
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file()

        # Override with environment variables
        self._load_from_env()

    def _load_from_file(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file) as f:
                config = json.load(f)

            # Update configuration sections
            for section_name, section_config in config.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_config.items():
                        if hasattr(section, key):
                            setattr(section, key, value)

        except Exception:
            pass

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Database configuration
        if os.getenv("ASTRA_DB_TYPE"):
            self.database.type = os.getenv("ASTRA_DB_TYPE")
        if os.getenv("ASTRA_DB_PATH"):
            self.database.path = os.getenv("ASTRA_DB_PATH")
        if os.getenv("ASTRA_DB_HOST"):
            self.database.host = os.getenv("ASTRA_DB_HOST")
        if os.getenv("ASTRA_DB_PORT"):
            self.database.port = int(os.getenv("ASTRA_DB_PORT"))

        # Data configuration
        if os.getenv("ASTRA_DATA_SOURCE"):
            self.data.primary_source = os.getenv("ASTRA_DATA_SOURCE")
        if os.getenv("ASTRA_CACHE_ENABLED"):
            self.data.cache_enabled = os.getenv("ASTRA_CACHE_ENABLED").lower() == "true"

        # Backtest configuration
        if os.getenv("ASTRA_INITIAL_CAPITAL"):
            self.backtest.initial_capital = float(os.getenv("ASTRA_INITIAL_CAPITAL"))
        if os.getenv("ASTRA_COMMISSION_RATE"):
            self.backtest.commission_rate = float(os.getenv("ASTRA_COMMISSION_RATE"))

        # Monte Carlo configuration
        if os.getenv("ASTRA_MC_SIMULATIONS"):
            self.monte_carlo.default_simulations = int(os.getenv("ASTRA_MC_SIMULATIONS"))
        if os.getenv("ASTRA_MC_USE_RUST"):
            self.monte_carlo.use_rust_engine = os.getenv("ASTRA_MC_USE_RUST").lower() == "true"

        # Risk configuration
        if os.getenv("ASTRA_MAX_DRAWDOWN"):
            self.risk.max_drawdown = float(os.getenv("ASTRA_MAX_DRAWDOWN"))
        if os.getenv("ASTRA_MAX_POSITION_WEIGHT"):
            self.risk.max_position_weight = float(os.getenv("ASTRA_MAX_POSITION_WEIGHT"))

    def save_config(self, filepath: str) -> None:
        """Save current configuration to file."""
        config = {
            "database": self.database.__dict__,
            "data": self.data.__dict__,
            "backtest": self.backtest.__dict__,
            "monte_carlo": self.monte_carlo.__dict__,
            "risk": self.risk.__dict__,
            "visualization": self.visualization.__dict__,
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    def get_results_dir(self) -> str:
        """Get results directory path."""
        return os.path.join(os.getcwd(), "results")

    def get_data_dir(self) -> str:
        """Get data directory path."""
        return os.path.join(os.getcwd(), "data")

    def get_config_dir(self) -> str:
        """Get config directory path."""
        return os.path.join(os.getcwd(), "config")

# Global settings instance
settings = Settings()
