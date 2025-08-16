#!/usr/bin/env python3
"""Quick Foundation Test - Astra Trading Platform.
==============================================

Test core functionality without external dependencies.
"""

import sys

sys.path.append("/mnt/f/astra/astra-main")

from typing import Optional

import numpy as np
import pandas as pd

# Test imports
try:
    from src.backtesting.engine import BacktestEngine, BacktestResult
    from src.backtesting.strategies import MeanReversionStrategy, MomentumStrategy
except ImportError:
    sys.exit(1)

def create_sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    np.random.seed(42)

    # Generate synthetic price data with trend
    n_days = len(dates)
    symbols = ["STOCK_A", "STOCK_B", "STOCK_C"]

    data = {}
    for symbol in symbols:
        # Random walk with slight upward drift
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = [100.0]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data[symbol] = prices

    return pd.DataFrame(data, index=dates)

def test_strategy_signals():
    """Test strategy signal generation."""
    data = create_sample_data()

    # Test momentum strategy
    momentum = MomentumStrategy(short_window=5, long_window=20)
    signals = momentum.generate_signals(data)


    # Test mean reversion strategy
    mean_rev = MeanReversionStrategy(window=20, threshold=1.5)
    mean_rev.generate_signals(data)


    return data, signals

def test_backtest_engine():
    """Test backtest engine."""
    data, signals = test_strategy_signals()

    # Run backtest
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    return engine.run_backtest(data, signals, position_size=0.5)



def test_monte_carlo_placeholder() -> Optional[bool]:
    """Test Monte Carlo integration placeholder."""
    try:
        # This will fail until we build the Rust module
        import astra_monte_carlo as mc
        return True
    except ImportError:
        return False

if __name__ == "__main__":

    try:
        # Test core functionality
        result = test_backtest_engine()

        # Test Monte Carlo (expected to fail for now)
        mc_available = test_monte_carlo_placeholder()



    except Exception:
        import traceback
        traceback.print_exc()
