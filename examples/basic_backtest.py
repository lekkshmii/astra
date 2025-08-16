#!/usr/bin/env python3
"""
Astra Trading Platform - Basic Backtest Example
================================================================

A simple momentum strategy backtest using VectorBT to demonstrate
the foundation of our trading system.

This is our Phase 1 milestone - getting VectorBT working with
basic strategies and market data.
"""

import sys
sys.path.append('/mnt/f/astra/astra-main')

from src.data.loader import DataLoader
from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import MomentumStrategy, MeanReversionStrategy
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def basic_momentum_backtest():
    """
    Run a basic momentum strategy backtest on major tech stocks.
    
    Strategy:
    - Buy when short MA > long MA
    - Sell when short MA <= long MA
    - Equal weight allocation
    
    Returns:
        BacktestResult: Backtest results
    """
    print("ASTRA TRADING PLATFORM - Basic Backtest")
    print("=" * 50)
    
    # Download data for major tech stocks
    loader = DataLoader()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"Downloading data for {symbols}")
    print(f"Period: {start_date} to {end_date}")
    
    try:
        data = loader.download_adjusted_close(symbols, start_date, end_date)
        print(f"Downloaded {len(data)} days of data")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Create momentum strategy
    strategy = MomentumStrategy(short_window=10, long_window=30)
    print(f"Strategy: {strategy}")
    
    # Generate signals
    print("Generating trading signals...")
    signals = strategy.generate_signals(data)
    
    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    result = engine.run_backtest(data, signals, position_size=0.8)
    
    # Display results
    print("\nBACKTEST RESULTS")
    print("=" * 30)
    print(f"Total Return:     {result.metrics['total_return']:.2%}")
    print(f"Annual Return:    {result.metrics['annual_return']:.2%}")
    print(f"Volatility:       {result.metrics['volatility']:.2%}")
    print(f"Sharpe Ratio:     {result.metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:     {result.metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:     {result.metrics['calmar_ratio']:.3f}")
    print(f"Final Value:      ${result.metrics['final_value']:,.2f}")
    
    # Show trade statistics
    if not result.trades.empty:
        print(f"\nTRADE STATISTICS")
        print("=" * 20)
        print(f"Total Trades:        {len(result.trades)}")
        profit_trades = result.trades[result.trades['value'] > 0]
        print(f"Profitable Trades:   {len(profit_trades)}")
        if len(result.trades) > 0:
            print(f"Win Rate:           {len(profit_trades)/len(result.trades):.1%}")
    
    return result

def plot_results(result):
    """Plot backtest results."""
    if result is None:
        return
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value over time
    result.portfolio_value.plot(ax=ax1, title='Astra Portfolio Value', 
                               color='#2E86AB', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and plot drawdown
    cumulative = (1 + result.returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    
    drawdown.plot(ax=ax2, title='Drawdown', 
                  color='#A23B72', linewidth=2)
    ax2.fill_between(drawdown.index, drawdown.values, 
                     alpha=0.3, color='#A23B72')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/f/astra/astra-main/results/plots/basic_backtest.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nChart saved to results/plots/basic_backtest.png")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('/mnt/f/astra/astra-main/results/plots', exist_ok=True)
    
    # Run backtest
    result = basic_momentum_backtest()
    
    # Plot results
    plot_results(result)
    
    print("\nPhase 1 Complete - Backtesting Foundation Working!")
    print("Next: Phase 2 - Rust Monte Carlo Integration")