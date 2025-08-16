#!/usr/bin/env python3
"""Astra Trading Platform - Basic Backtest Example.
================================================================

A simple momentum strategy backtest using VectorBT to demonstrate
the foundation of our trading system.

This is our Phase 1 milestone - getting VectorBT working with
basic strategies and market data.
"""

import sys

sys.path.append("/mnt/f/astra/astra-main")

import matplotlib.pyplot as plt

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import MomentumStrategy
from src.data.loader import DataLoader


def basic_momentum_backtest():
    """Run a basic momentum strategy backtest on major tech stocks.

    Strategy:
    - Buy when short MA > long MA
    - Sell when short MA <= long MA
    - Equal weight allocation

    Returns
    -------
        BacktestResult: Backtest results

    """
    # Download data for major tech stocks
    loader = DataLoader()
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2024-01-01"


    try:
        data = loader.download_adjusted_close(symbols, start_date, end_date)
    except Exception:
        return None

    # Create momentum strategy
    strategy = MomentumStrategy(short_window=10, long_window=30)

    # Generate signals
    signals = strategy.generate_signals(data)

    # Run backtest
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    result = engine.run_backtest(data, signals, position_size=0.8)

    # Display results

    # Show trade statistics
    if not result.trades.empty:
        result.trades[result.trades["value"] > 0]
        if len(result.trades) > 0:
            pass

    return result

def plot_results(result) -> None:
    """Plot backtest results."""
    if result is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value over time
    result.portfolio_value.plot(ax=ax1, title="Astra Portfolio Value",
                               color="#2E86AB", linewidth=2)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)

    # Calculate and plot drawdown
    cumulative = (1 + result.returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak

    drawdown.plot(ax=ax2, title="Drawdown",
                  color="#A23B72", linewidth=2)
    ax2.fill_between(drawdown.index, drawdown.values,
                     alpha=0.3, color="#A23B72")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/f/astra/astra-main/results/plots/basic_backtest.png",
                dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs("/mnt/f/astra/astra-main/results/plots", exist_ok=True)

    # Run backtest
    result = basic_momentum_backtest()

    # Plot results
    plot_results(result)

