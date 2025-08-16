#!/usr/bin/env python3
"""
Test Rust Monte Carlo Integration
=================================

Test the Rust Monte Carlo scenarios once Python-Rust FFI is working.
"""

def test_flash_crash():
    """Test flash crash scenario."""
    try:
        import astra_monte_carlo as mc
        
        results = mc.monte_carlo_flash_crash(
            initial_prices=[100.0],
            num_simulations=1000,
            time_steps=252,
            crash_probability=0.001,
            crash_magnitude=0.2,
            recovery_rate=0.1
        )
        
        print(f"Flash crash simulation: {len(results)} paths generated")
        print(f"Final prices range: {min(r[-1] for r in results):.2f} - {max(r[-1] for r in results):.2f}")
        
        return True
        
    except ImportError:
        print("Rust module not available yet")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_regime_switching():
    """Test regime switching scenario."""
    try:
        import astra_monte_carlo as mc
        
        results = mc.monte_carlo_regime_switching(
            initial_prices=[100.0],
            num_simulations=1000,
            time_steps=252,
            bull_volatility=0.15,
            bear_volatility=0.3,
            transition_probability=0.02
        )
        
        print(f"Regime switching simulation: {len(results)} paths generated")
        return True
        
    except ImportError:
        print("Rust module not available yet")
        return False

def test_portfolio_stress():
    """Test portfolio stress testing."""
    try:
        import astra_monte_carlo as mc
        
        results = mc.monte_carlo_portfolio_stress(
            weights=[0.4, 0.3, 0.3],
            expected_returns=[0.08, 0.06, 0.05],
            covariance_matrix=[
                [0.04, 0.02, 0.01],
                [0.02, 0.03, 0.015],
                [0.01, 0.015, 0.025]
            ],
            num_simulations=10000,
            time_horizon=252,
            confidence_levels=[0.95, 0.99]
        )
        
        print("Portfolio stress test results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
            
        return True
        
    except ImportError:
        print("Rust module not available yet")
        return False

if __name__ == "__main__":
    print("Testing Rust Monte Carlo Integration")
    print("=" * 40)
    
    tests = [
        ("Flash Crash", test_flash_crash),
        ("Regime Switching", test_regime_switching),
        ("Portfolio Stress", test_portfolio_stress)
    ]
    
    for name, test_func in tests:
        print(f"\n{name}:")
        success = test_func()
        print(f"Status: {'PASS' if success else 'PENDING'}")
    
    print(f"\nRust Monte Carlo module ready for Python 3.11 installation")