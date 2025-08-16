#!/usr/bin/env python3
"""Risk Management Demo - Astra Trading Platform.
=============================================

Comprehensive demonstration of institutional risk management capabilities.
"""

import sys

sys.path.append("/mnt/f/astra/astra-main")

import numpy as np
import pandas as pd

from src.risk_management.circuit_breakers import CircuitBreakerManager
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.risk_monitor import RiskMonitor


def demo_circuit_breakers():
    """Demonstrate circuit breaker functionality."""
    # Create circuit breaker manager
    cb_manager = CircuitBreakerManager()

    # Simulate portfolio data with drawdown
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    portfolio_values = [100000]

    # Simulate a drawdown scenario
    for i in range(1, 100):
        if i < 50:
            # Normal volatility
            daily_return = np.random.normal(0.001, 0.015)
        else:
            # Stress period with drawdown
            daily_return = np.random.normal(-0.002, 0.035)

        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    returns_series = portfolio_series.pct_change().dropna()

    # Test circuit breakers
    market_data = {
        "portfolio_value": portfolio_series,
        "returns": returns_series,
        "position_weights": {"AAPL": 0.35, "MSFT": 0.25, "GOOGL": 0.20, "AMZN": 0.20},
    }

    events = cb_manager.check_all_breakers(**market_data)


    if events:
        for _event in events:
            pass
    else:
        pass

    # Show trading status
    cb_manager.get_trading_status()

    return cb_manager, portfolio_series, returns_series

def demo_position_sizing():
    """Demonstrate position sizing methods."""
    # Setup position sizer
    position_sizer = PositionSizer()

    # Sample data
    portfolio_value = 100000
    signals = pd.Series({
        "AAPL": 1.0,   # Long signal
        "MSFT": 1.0,   # Long signal
        "GOOGL": -1.0, # Short signal
        "AMZN": 0.0,   # No signal
        "TSLA": 1.0,    # Long signal
    })

    prices = pd.Series({
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOGL": 2500.0,
        "AMZN": 180.0,
        "TSLA": 200.0,
    })

    # Generate sample returns data for advanced methods
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    returns_data = pd.DataFrame({
        symbol: np.random.normal(0.001, 0.02, 50)
        for symbol in signals.index
    }, index=dates)

    # Test different sizing methods
    methods = ["FixedFractional", "VolatilityTarget", "KellyCriterion"]

    for method in methods:

        kwargs = {"returns_data": returns_data} if method != "FixedFractional" else {}
        positions = position_sizer.size_positions(
            portfolio_value, signals, prices, method=method, **kwargs,
        )

        for _symbol, _pos in positions.items():
            pass

        # Portfolio summary
        position_sizer.get_portfolio_summary(positions)

    return position_sizer

def demo_risk_monitoring():
    """Demonstrate real-time risk monitoring."""
    # Create risk monitor
    risk_monitor = RiskMonitor()

    # Simulate portfolio evolution
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    portfolio_values = []
    returns = []

    initial_value = 100000
    current_value = initial_value

    for i in range(30):
        if i < 20:
            daily_return = np.random.normal(0.001, 0.015)
        else:
            # Stress period
            daily_return = np.random.normal(-0.01, 0.04)

        current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
        returns.append(daily_return)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    returns_series = pd.Series(returns, index=dates)

    # Current positions
    positions = {
        "AAPL": 0.30,
        "MSFT": 0.25,
        "GOOGL": 0.20,
        "AMZN": 0.15,
        "TSLA": 0.10,
    }

    # Portfolio data for risk monitoring
    portfolio_data = {
        "portfolio_value": current_value,
        "initial_value": initial_value,
        "returns": returns_series,
        "value_series": portfolio_series,
        "positions": positions,
        "prices": pd.Series({"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "AMZN": 180, "TSLA": 200}),
    }

    # Run comprehensive risk check
    risk_assessment = risk_monitor.run_comprehensive_risk_check(portfolio_data)

    # Display results
    risk_assessment["risk_metrics"]

    # Show alerts
    alerts = risk_assessment["alerts"]
    if alerts:
        for _alert in alerts:
            pass
    else:
        pass

    # Show risk score and recommendations

    for _rec in risk_assessment["recommendations"]:
        pass

    return risk_monitor

def demo_integrated_risk_system():
    """Demonstrate integrated risk management system."""
    # Create integrated system
    cb_manager = CircuitBreakerManager()
    position_sizer = PositionSizer()
    risk_monitor = RiskMonitor()

    # Simulate trading day with risk events
    portfolio_value = 100000

    # Morning: Normal trading
    signals = pd.Series({"AAPL": 1.0, "MSFT": 1.0, "GOOGL": 0.5})
    prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0})

    positions = position_sizer.size_positions(portfolio_value, signals, prices)

    # Midday: Volatility spike

    # Simulate high volatility returns
    volatile_returns = pd.Series([0.05, -0.08, 0.06, -0.04, 0.03])

    events = cb_manager.check_all_breakers(returns=volatile_returns)
    if events:
        pass

    # Afternoon: Risk reassessment

    # Update portfolio after volatility
    new_portfolio_value = portfolio_value * 0.92  # 8% drawdown

    portfolio_data = {
        "portfolio_value": new_portfolio_value,
        "initial_value": portfolio_value,
        "returns": volatile_returns,
        "value_series": pd.Series([portfolio_value, new_portfolio_value]),
        "positions": {pos.symbol: pos.target_weight for pos in positions.values()},
    }

    risk_assessment = risk_monitor.run_comprehensive_risk_check(portfolio_data)


    if risk_assessment["alerts"]:
        for _alert in risk_assessment["alerts"][:2]:  # Show first 2
            pass

    # End of day: Status summary
    cb_manager.get_trading_status()
    risk_monitor.get_risk_dashboard()



    return {
        "circuit_breakers": cb_manager,
        "position_sizer": position_sizer,
        "risk_monitor": risk_monitor,
    }

if __name__ == "__main__":

    try:
        # Run individual component demos
        cb_manager, portfolio_series, returns_series = demo_circuit_breakers()
        position_sizer = demo_position_sizing()
        risk_monitor = demo_risk_monitoring()

        # Run integrated system demo
        integrated_system = demo_integrated_risk_system()



    except Exception:
        import traceback
        traceback.print_exc()
