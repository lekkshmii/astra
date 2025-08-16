#!/usr/bin/env python3
"""Test Rust Monte Carlo Integration.
=================================

Test the Rust Monte Carlo scenarios once Python-Rust FFI is working.
"""
from typing import Optional


def test_flash_crash() -> Optional[bool]:
    """Test flash crash scenario."""
    try:
        import astra_monte_carlo as mc

        mc.monte_carlo_flash_crash(
            initial_prices=[100.0],
            num_simulations=1000,
            time_steps=252,
            crash_probability=0.001,
            crash_magnitude=0.2,
            recovery_rate=0.1,
        )


        return True

    except ImportError:
        return False
    except Exception:
        return False

def test_regime_switching() -> Optional[bool]:
    """Test regime switching scenario."""
    try:
        import astra_monte_carlo as mc

        mc.monte_carlo_regime_switching(
            initial_prices=[100.0],
            num_simulations=1000,
            time_steps=252,
            bull_volatility=0.15,
            bear_volatility=0.3,
            transition_probability=0.02,
        )

        return True

    except ImportError:
        return False

def test_portfolio_stress() -> Optional[bool]:
    """Test portfolio stress testing."""
    try:
        import astra_monte_carlo as mc

        results = mc.monte_carlo_portfolio_stress(
            weights=[0.4, 0.3, 0.3],
            expected_returns=[0.08, 0.06, 0.05],
            covariance_matrix=[
                [0.04, 0.02, 0.01],
                [0.02, 0.03, 0.015],
                [0.01, 0.015, 0.025],
            ],
            num_simulations=10000,
            time_horizon=252,
            confidence_levels=[0.95, 0.99],
        )

        for _key, _value in results.items():
            pass

        return True

    except ImportError:
        return False

if __name__ == "__main__":

    tests = [
        ("Flash Crash", test_flash_crash),
        ("Regime Switching", test_regime_switching),
        ("Portfolio Stress", test_portfolio_stress),
    ]

    for _name, test_func in tests:
        success = test_func()

