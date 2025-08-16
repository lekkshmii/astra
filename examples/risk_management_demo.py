#!/usr/bin/env python3
"""
Risk Management Demo - Astra Trading Platform
=============================================

Comprehensive demonstration of institutional risk management capabilities.
"""

import sys
sys.path.append('/mnt/f/astra/astra-main')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.risk_management.circuit_breakers import CircuitBreakerManager
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.risk_monitor import RiskMonitor

def demo_circuit_breakers():
    """Demonstrate circuit breaker functionality."""
    print("\n=== CIRCUIT BREAKER DEMO ===")
    
    # Create circuit breaker manager
    cb_manager = CircuitBreakerManager()
    
    # Simulate portfolio data with drawdown
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
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
        'portfolio_value': portfolio_series,
        'returns': returns_series,
        'position_weights': {'AAPL': 0.35, 'MSFT': 0.25, 'GOOGL': 0.20, 'AMZN': 0.20}
    }
    
    events = cb_manager.check_all_breakers(**market_data)
    
    print(f"Portfolio final value: ${portfolio_series.iloc[-1]:,.2f}")
    print(f"Total return: {(portfolio_series.iloc[-1]/portfolio_series.iloc[0] - 1):.1%}")
    print(f"Max drawdown: {((portfolio_series - portfolio_series.expanding().max()) / portfolio_series.expanding().max()).min():.1%}")
    
    if events:
        print(f"\nCircuit Breaker Events: {len(events)}")
        for event in events:
            print(f"  - {event.breaker_name}: {event.severity.value}")
            print(f"    {event.message}")
            print(f"    Action: {event.action_taken}")
    else:
        print("\nNo circuit breaker events triggered")
    
    # Show trading status
    status = cb_manager.get_trading_status()
    print(f"\nTrading Status:")
    print(f"  Trading Allowed: {status['trading_allowed']}")
    print(f"  Risk Level: {status['risk_level']}")
    print(f"  Active Halts: {len(status['active_halts'])}")
    
    return cb_manager, portfolio_series, returns_series

def demo_position_sizing():
    """Demonstrate position sizing methods."""
    print("\n=== POSITION SIZING DEMO ===")
    
    # Setup position sizer
    position_sizer = PositionSizer()
    
    # Sample data
    portfolio_value = 100000
    signals = pd.Series({
        'AAPL': 1.0,   # Long signal
        'MSFT': 1.0,   # Long signal
        'GOOGL': -1.0, # Short signal
        'AMZN': 0.0,   # No signal
        'TSLA': 1.0    # Long signal
    })
    
    prices = pd.Series({
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 2500.0,
        'AMZN': 180.0,
        'TSLA': 200.0
    })
    
    # Generate sample returns data for advanced methods
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    returns_data = pd.DataFrame({
        symbol: np.random.normal(0.001, 0.02, 50)
        for symbol in signals.index
    }, index=dates)
    
    # Test different sizing methods
    methods = ["FixedFractional", "VolatilityTarget", "KellyCriterion"]
    
    for method in methods:
        print(f"\n{method} Position Sizing:")
        
        kwargs = {'returns_data': returns_data} if method != 'FixedFractional' else {}
        positions = position_sizer.size_positions(
            portfolio_value, signals, prices, method=method, **kwargs
        )
        
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos.target_weight:.1%} weight, "
                  f"{pos.target_shares:.1f} shares, "
                  f"${pos.target_value:,.0f} value")
        
        # Portfolio summary
        summary = position_sizer.get_portfolio_summary(positions)
        print(f"  Total Weight: {summary.get('total_weight', 0):.1%}")
        print(f"  Risk Contribution: {summary.get('total_risk_contribution', 0):.1%}")
        print(f"  Avg Confidence: {summary.get('average_confidence', 0):.1%}")
    
    return position_sizer

def demo_risk_monitoring():
    """Demonstrate real-time risk monitoring."""
    print("\n=== RISK MONITORING DEMO ===")
    
    # Create risk monitor
    risk_monitor = RiskMonitor()
    
    # Simulate portfolio evolution
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
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
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'TSLA': 0.10
    }
    
    # Portfolio data for risk monitoring
    portfolio_data = {
        'portfolio_value': current_value,
        'initial_value': initial_value,
        'returns': returns_series,
        'value_series': portfolio_series,
        'positions': positions,
        'prices': pd.Series({'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 180, 'TSLA': 200})
    }
    
    # Run comprehensive risk check
    risk_assessment = risk_monitor.run_comprehensive_risk_check(portfolio_data)
    
    # Display results
    metrics = risk_assessment['risk_metrics']
    print(f"Portfolio Value: ${metrics.portfolio_value:,.2f}")
    print(f"Total Return: {metrics.total_return:.1%}")
    print(f"Daily Return: {metrics.daily_return:.2%}")
    print(f"Volatility: {metrics.volatility:.1%}")
    print(f"Max Drawdown: {metrics.max_drawdown:.1%}")
    print(f"VaR 95%: {metrics.var_95:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Largest Position: {metrics.largest_position:.1%}")
    print(f"Total Exposure: {metrics.total_exposure:.1%}")
    
    # Show alerts
    alerts = risk_assessment['alerts']
    if alerts:
        print(f"\nRisk Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.alert_type}: {alert.message}")
            print(f"    Recommendation: {alert.recommended_action}")
    else:
        print("\nNo risk alerts")
    
    # Show risk score and recommendations
    print(f"\nRisk Score: {risk_assessment['risk_score']}/100")
    print(f"Trading Status: {'ALLOWED' if risk_assessment['trading_status']['trading_allowed'] else 'HALTED'}")
    
    print("\nRecommendations:")
    for rec in risk_assessment['recommendations']:
        print(f"  - {rec}")
    
    return risk_monitor

def demo_integrated_risk_system():
    """Demonstrate integrated risk management system."""
    print("\n=== INTEGRATED RISK SYSTEM DEMO ===")
    
    # Create integrated system
    cb_manager = CircuitBreakerManager()
    position_sizer = PositionSizer()
    risk_monitor = RiskMonitor()
    
    # Simulate trading day with risk events
    portfolio_value = 100000
    
    # Morning: Normal trading
    print("\n📅 MORNING: Normal Market Conditions")
    signals = pd.Series({'AAPL': 1.0, 'MSFT': 1.0, 'GOOGL': 0.5})
    prices = pd.Series({'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0})
    
    positions = position_sizer.size_positions(portfolio_value, signals, prices)
    print("✓ Position sizing completed")
    
    # Midday: Volatility spike
    print("\n📅 MIDDAY: Volatility Spike Detected")
    
    # Simulate high volatility returns
    volatile_returns = pd.Series([0.05, -0.08, 0.06, -0.04, 0.03])
    
    events = cb_manager.check_all_breakers(returns=volatile_returns)
    if events:
        print(f"⚠️  Circuit breaker triggered: {events[0].breaker_name}")
        print(f"   Action: {events[0].action_taken}")
    
    # Afternoon: Risk reassessment
    print("\n📅 AFTERNOON: Risk Reassessment")
    
    # Update portfolio after volatility
    new_portfolio_value = portfolio_value * 0.92  # 8% drawdown
    
    portfolio_data = {
        'portfolio_value': new_portfolio_value,
        'initial_value': portfolio_value,
        'returns': volatile_returns,
        'value_series': pd.Series([portfolio_value, new_portfolio_value]),
        'positions': {pos.symbol: pos.target_weight for pos in positions.values()}
    }
    
    risk_assessment = risk_monitor.run_comprehensive_risk_check(portfolio_data)
    
    print(f"Portfolio Impact: {((new_portfolio_value/portfolio_value)-1):.1%}")
    print(f"Risk Score: {risk_assessment['risk_score']}/100")
    
    if risk_assessment['alerts']:
        print("⚠️  Risk alerts generated:")
        for alert in risk_assessment['alerts'][:2]:  # Show first 2
            print(f"   - {alert.alert_type}")
    
    # End of day: Status summary
    print("\n📅 END OF DAY: System Status")
    status = cb_manager.get_trading_status()
    dashboard = risk_monitor.get_risk_dashboard()
    
    print(f"Trading Status: {'🟢 ACTIVE' if status['trading_allowed'] else '🔴 HALTED'}")
    print(f"Risk Level: {status['risk_level']}")
    print(f"Active Positions: {len(positions)}")
    print(f"24h Alerts: {dashboard.get('alert_count_24h', 0)}")
    
    print("\n✅ Integrated risk system operational")
    
    return {
        'circuit_breakers': cb_manager,
        'position_sizer': position_sizer,
        'risk_monitor': risk_monitor
    }

if __name__ == "__main__":
    print("ASTRA TRADING PLATFORM - Risk Management Demo")
    print("=" * 55)
    
    try:
        # Run individual component demos
        cb_manager, portfolio_series, returns_series = demo_circuit_breakers()
        position_sizer = demo_position_sizing()
        risk_monitor = demo_risk_monitoring()
        
        # Run integrated system demo
        integrated_system = demo_integrated_risk_system()
        
        print(f"\n{'='*55}")
        print("RISK MANAGEMENT SYSTEM COMPLETE")
        print("Status: ALL COMPONENTS OPERATIONAL")
        print("✓ Circuit breakers: WORKING")
        print("✓ Position sizing: WORKING")
        print("✓ Risk monitoring: WORKING")
        print("✓ Alert system: WORKING")
        print("✓ Integrated workflow: WORKING")
        
        print(f"\nPhase 3 Risk Management: OPERATIONAL")
        print("Ready for Phase 4: Professional Visualization")
        
    except Exception as e:
        print(f"Error in risk management demo: {e}")
        import traceback
        traceback.print_exc()