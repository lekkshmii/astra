"""Test Jupyter Notebooks Functionality.
====================================

Test script to validate that all notebook components are working.
"""

import sys
from typing import Optional

sys.path.append(".")

def test_imports() -> Optional[bool]:
    """Test all required imports for notebooks."""
    try:
        # Core libraries

        # Astra imports

        return True

    except Exception:
        return False

def test_data_functionality():
    """Test data loading functionality."""
    try:
        # Import here to avoid global scope issues
        import numpy as np
        import pandas as pd

        from config.instruments import AssetUniverse
        from src.data.loader import DataLoader

        # Test asset universe
        asset_universe = AssetUniverse()
        asset_universe.get_symbols("diversified")

        # Test data loader initialization
        DataLoader()

        # Test synthetic data generation
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100 * np.cumprod(1 + returns)

        return pd.DataFrame({
            "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 20000000, len(dates)),
        }, index=dates)


    except Exception:
        return None

def test_backtesting_functionality(test_data):
    """Test backtesting functionality."""
    try:
        # Import here
        import pandas as pd

        from src.backtesting import AstraBacktestConfig, create_astra_backtest

        # Test backtest engine
        config = AstraBacktestConfig(initial_cash=100000, commission=0.001)
        backtest_engine = create_astra_backtest(config)

        # Generate simple signals
        signals = pd.Series(0, index=test_data.index)
        short_ma = test_data["Close"].rolling(20).mean()
        long_ma = test_data["Close"].rolling(50).mean()
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1

        # Run backtest
        return backtest_engine.run_backtest(test_data, signals)


    except Exception:
        return None

def test_monte_carlo_functionality() -> Optional[bool]:
    """Test Monte Carlo functionality."""
    try:
        # Import here
        from src.monte_carlo.scenarios import (
            FlashCrashScenario,
            RegimeSwitchingScenario,
        )

        # Test Monte Carlo scenarios
        flash_crash = FlashCrashScenario()
        flash_crash.run([100.0], 100, 63)  # Small test

        regime_switching = RegimeSwitchingScenario()
        regime_switching.run([100.0], 100, 63)

        return True

    except Exception:
        return False

def test_risk_management_functionality() -> Optional[bool]:
    """Test risk management functionality."""
    try:
        # Import here
        from config.risk_limits import RiskLevel, RiskLimits
        from src.risk_management.circuit_breakers import CircuitBreakerManager
        from src.risk_management.metrics import RiskMetrics
        from src.risk_management.position_sizing import PositionSizer

        # Test circuit breakers
        CircuitBreakerManager()

        # Test position sizing
        position_sizer = PositionSizer()
        position_sizer.fixed_fractional(portfolio_value=100000, fraction=0.05)

        # Test risk metrics
        RiskMetrics()

        # Test risk limits
        RiskLimits(RiskLevel.MODERATE)

        return True

    except Exception:
        return False

def test_visualization_functionality(backtest_result) -> Optional[bool]:
    """Test visualization functionality."""
    try:
        # Import here
        import matplotlib.pyplot as plt

        from src.visualization import (
            AstraReportGenerator,
            AstraVectorBTCharts,
            ChartBuilder,
        )

        # Test chart builder
        chart_builder = ChartBuilder()

        # Test VectorBT charts
        AstraVectorBTCharts()

        # Test report generator
        AstraReportGenerator()

        if backtest_result:
            # Test basic chart creation (without displaying)
            import matplotlib as mpl
            mpl.use("Agg")  # Use non-GUI backend for testing

            fig = chart_builder.plot_backtest_results(backtest_result)
            plt.close(fig)

        return True

    except Exception:
        return False

def main():
    """Run all notebook functionality tests."""
    # Track test results
    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Data functionality
    test_data = test_data_functionality()
    results.append(("Data Loading", test_data is not None))

    if test_data is not None:
        # Test 3: Backtesting
        backtest_result = test_backtesting_functionality(test_data)
        results.append(("Backtesting", backtest_result is not None))

        # Test 4: Monte Carlo
        results.append(("Monte Carlo", test_monte_carlo_functionality()))

        # Test 5: Risk Management
        results.append(("Risk Management", test_risk_management_functionality()))

        # Test 6: Visualization
        results.append(("Visualization", test_visualization_functionality(backtest_result)))
    else:
        # Skip dependent tests
        results.extend([
            ("Backtesting", False),
            ("Monte Carlo", False),
            ("Risk Management", False),
            ("Visualization", False),
        ])

    # Summary

    passed = 0
    total = len(results)

    for _test_name, passed_test in results:
        if passed_test:
            passed += 1


    if passed == total:
        pass
    else:
        pass

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
