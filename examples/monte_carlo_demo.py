#!/usr/bin/env python3
"""
Monte Carlo Demo - Astra Trading Platform
=========================================

Comprehensive demonstration of Monte Carlo risk scenarios.
"""

import sys
sys.path.append('/mnt/f/astra/astra-main')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.monte_carlo.scenarios import (
    ScenarioRunner, FlashCrashScenario, 
    RegimeSwitchingScenario, CorrelationBreakdownScenario
)

def demo_flash_crash():
    """Demonstrate flash crash scenario."""
    print("\n=== FLASH CRASH SCENARIO ===")
    
    scenario = FlashCrashScenario(
        crash_probability=0.005,  # Higher probability for demo
        crash_magnitude=0.20,     # 20% crash
        recovery_rate=0.1
    )
    
    result = scenario.run(
        initial_prices=[100.0],
        num_simulations=1000,     # Smaller for demo
        time_steps=252
    )
    
    print(f"Crash frequency: {result.statistics['crash_frequency']:.1%}")
    print(f"Mean final price: ${result.statistics['mean_final_price']:.2f}")
    print(f"Price volatility: ${result.statistics['std_final_price']:.2f}")
    print(f"Mean max drawdown: {result.statistics['max_drawdown_mean']:.1%}")
    
    print("\nPrice Distribution:")
    print(f"  5th percentile:  ${result.percentiles['p5']:.2f}")
    print(f"  25th percentile: ${result.percentiles['p25']:.2f}")
    print(f"  50th percentile: ${result.percentiles['p50']:.2f}")
    print(f"  75th percentile: ${result.percentiles['p75']:.2f}")
    print(f"  95th percentile: ${result.percentiles['p95']:.2f}")
    
    return result

def demo_regime_switching():
    """Demonstrate regime switching scenario."""
    print("\n=== REGIME SWITCHING SCENARIO ===")
    
    scenario = RegimeSwitchingScenario(
        bull_volatility=0.15,      # 15% bull market volatility
        bear_volatility=0.35,      # 35% bear market volatility
        transition_probability=0.03 # 3% daily transition probability
    )
    
    result = scenario.run(
        initial_prices=[100.0],
        num_simulations=1000,
        time_steps=252
    )
    
    print(f"Mean final price: ${result.statistics['mean_final_price']:.2f}")
    print(f"Price volatility: ${result.statistics['std_final_price']:.2f}")
    print(f"Expected transitions: {result.statistics['regime_transitions_mean']:.1f}")
    
    print("\nPrice Distribution:")
    print(f"  5th percentile:  ${result.percentiles['p5']:.2f}")
    print(f"  95th percentile: ${result.percentiles['p95']:.2f}")
    print(f"  Range: {(result.percentiles['p95'] - result.percentiles['p5']):.2f}")
    
    return result

def demo_correlation_breakdown():
    """Demonstrate correlation breakdown scenario."""
    print("\n=== CORRELATION BREAKDOWN SCENARIO ===")
    
    scenario = CorrelationBreakdownScenario(
        normal_correlation=0.3,    # 30% normal correlation
        stress_correlation=0.8,    # 80% stress correlation
        stress_threshold=2.0       # 2-sigma stress threshold
    )
    
    # Multi-asset portfolio
    initial_prices = [100.0, 150.0, 200.0, 80.0]
    
    result = scenario.run(
        initial_prices=initial_prices,
        num_simulations=1000,
        time_steps=252
    )
    
    print("Final Price Statistics:")
    for i, price in enumerate(result.statistics['mean_final_prices']):
        print(f"  Asset {i+1}: ${price:.2f} (started at ${initial_prices[i]:.2f})")
    
    print(f"\nStress correlation frequency: {result.statistics['correlation_stress_frequency']:.1%}")
    
    return result

def demo_portfolio_stress_test():
    """Demonstrate comprehensive portfolio stress testing."""
    print("\n=== PORTFOLIO STRESS TEST ===")
    
    # Tech portfolio example
    portfolio_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weight
    asset_prices = [150.0, 300.0, 2500.0, 180.0]   # AAPL, MSFT, GOOGL, AMZN style
    
    runner = ScenarioRunner()
    
    stress_results = runner.portfolio_stress_test(
        portfolio_weights=portfolio_weights,
        asset_prices=asset_prices,
        scenarios=['flash_crash', 'regime_switching']
    )
    
    print("Portfolio Stress Test Results:")
    print("=" * 40)
    
    for scenario_name, metrics in stress_results.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        print(f"  Volatility:      {metrics['volatility']:.2%}")
        print(f"  VaR 95%:         {metrics['var_95']:.2%}")
        print(f"  VaR 99%:         {metrics['var_99']:.2%}")
        print(f"  Max Loss:        {metrics['max_loss']:.2%}")
    
    return stress_results

def plot_scenarios(flash_result, regime_result):
    """Plot scenario results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Flash crash paths (first 50 simulations)
    for i in range(min(50, len(flash_result.paths))):
        ax1.plot(flash_result.paths[i], alpha=0.3, color='red', linewidth=0.5)
    ax1.set_title('Flash Crash Scenarios (50 paths)')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Flash crash final price distribution
    final_prices_flash = flash_result.paths[:, -1]
    ax2.hist(final_prices_flash, bins=50, alpha=0.7, color='red', density=True)
    ax2.axvline(np.mean(final_prices_flash), color='black', linestyle='--', 
                label=f'Mean: ${np.mean(final_prices_flash):.2f}')
    ax2.set_title('Flash Crash Final Price Distribution')
    ax2.set_xlabel('Final Price')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Regime switching paths
    for i in range(min(50, len(regime_result.paths))):
        ax3.plot(regime_result.paths[i], alpha=0.3, color='blue', linewidth=0.5)
    ax3.set_title('Regime Switching Scenarios (50 paths)')
    ax3.set_ylabel('Price')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    
    # Regime switching final price distribution
    final_prices_regime = regime_result.paths[:, -1]
    ax4.hist(final_prices_regime, bins=50, alpha=0.7, color='blue', density=True)
    ax4.axvline(np.mean(final_prices_regime), color='black', linestyle='--',
                label=f'Mean: ${np.mean(final_prices_regime):.2f}')
    ax4.set_title('Regime Switching Final Price Distribution')
    ax4.set_xlabel('Final Price')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/f/astra/astra-main/results/plots/monte_carlo_scenarios.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nCharts saved to results/plots/monte_carlo_scenarios.png")

if __name__ == "__main__":
    print("ASTRA TRADING PLATFORM - Monte Carlo Demo")
    print("=" * 50)
    
    # Create results directory
    import os
    os.makedirs('/mnt/f/astra/astra-main/results/plots', exist_ok=True)
    
    try:
        # Run individual scenarios
        flash_result = demo_flash_crash()
        regime_result = demo_regime_switching()
        correlation_result = demo_correlation_breakdown()
        
        # Run portfolio stress test
        stress_results = demo_portfolio_stress_test()
        
        # Create visualizations
        plot_scenarios(flash_result, regime_result)
        
        print(f"\n{'='*50}")
        print("MONTE CARLO ANALYSIS COMPLETE")
        print("Status: ALL SCENARIOS OPERATIONAL")
        print("✓ Flash crash modeling: WORKING")
        print("✓ Regime switching: WORKING") 
        print("✓ Correlation breakdown: WORKING")
        print("✓ Portfolio stress testing: WORKING")
        print("✓ Risk visualization: WORKING")
        
        print(f"\nPhase 2 Monte Carlo Engine: OPERATIONAL")
        print("Ready for Phase 3: Risk Management Integration")
        
    except Exception as e:
        print(f"Error in Monte Carlo demo: {e}")
        import traceback
        traceback.print_exc()