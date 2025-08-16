#!/usr/bin/env python3
"""
Quick Foundation Test - Astra Trading Platform
==============================================

Test core functionality without external dependencies.
"""

import sys
sys.path.append('/mnt/f/astra/astra-main')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test imports
try:
    from src.backtesting.engine import BacktestEngine, BacktestResult
    from src.backtesting.strategies import MomentumStrategy, MeanReversionStrategy
    print("✓ Core modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def create_sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data with trend
    n_days = len(dates)
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
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
    print("\n=== Testing Strategy Signals ===")
    
    data = create_sample_data()
    print(f"Sample data shape: {data.shape}")
    
    # Test momentum strategy
    momentum = MomentumStrategy(short_window=5, long_window=20)
    signals = momentum.generate_signals(data)
    
    print(f"Momentum signals shape: {signals.shape}")
    print(f"Signal range: {signals.min().min():.1f} to {signals.max().max():.1f}")
    
    # Test mean reversion strategy
    mean_rev = MeanReversionStrategy(window=20, threshold=1.5)
    signals_mr = mean_rev.generate_signals(data)
    
    print(f"Mean reversion signals shape: {signals_mr.shape}")
    print("✓ Strategy signal generation working")
    
    return data, signals

def test_backtest_engine():
    """Test backtest engine."""
    print("\n=== Testing Backtest Engine ===")
    
    data, signals = test_strategy_signals()
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    result = engine.run_backtest(data, signals, position_size=0.5)
    
    print(f"Portfolio final value: ${result.metrics['final_value']:,.2f}")
    print(f"Total return: {result.metrics['total_return']:.2%}")
    print(f"Max drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.3f}")
    
    print("✓ Backtest engine working")
    return result

def test_monte_carlo_placeholder():
    """Test Monte Carlo integration placeholder."""
    print("\n=== Testing Monte Carlo Integration ===")
    
    try:
        # This will fail until we build the Rust module
        import astra_monte_carlo as mc
        print("✓ Rust Monte Carlo module available")
        return True
    except ImportError:
        print("○ Rust Monte Carlo module not built yet")
        return False

if __name__ == "__main__":
    print("ASTRA TRADING PLATFORM - Foundation Test")
    print("=" * 50)
    
    try:
        # Test core functionality
        result = test_backtest_engine()
        
        # Test Monte Carlo (expected to fail for now)
        mc_available = test_monte_carlo_placeholder()
        
        print(f"\n=== Foundation Status ===")
        print("✓ Data structures: WORKING")
        print("✓ Strategy engine: WORKING") 
        print("✓ Backtest engine: WORKING")
        print("✓ Performance metrics: WORKING")
        print(f"{'✓' if mc_available else '○'} Monte Carlo engine: {'WORKING' if mc_available else 'PENDING'}")
        
        print(f"\n🎉 Phase 1 Foundation: OPERATIONAL")
        print("Ready for Phase 2: Monte Carlo integration")
        
    except Exception as e:
        print(f"✗ Foundation test failed: {e}")
        import traceback
        traceback.print_exc()