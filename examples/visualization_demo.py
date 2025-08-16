"""Astra Visualization Demo.
========================

Demonstrate the comprehensive visualization capabilities of Astra Trading Platform.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import contextlib
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting import AstraBacktestConfig, create_astra_backtest

# Astra imports
from src.data.loader import DataLoader
from src.monte_carlo.scenarios import FlashCrashScenario
from src.visualization import AstraReportGenerator, AstraVectorBTCharts, ChartBuilder

# from src.risk_management.monitor import RiskMonitor  # Optional import

def main() -> None:
    """Run comprehensive visualization demonstration."""
    # Initialize components
    data_loader = DataLoader()
    chart_builder = ChartBuilder(style="professional")
    vectorbt_charts = AstraVectorBTCharts(theme="astra_professional")
    report_generator = AstraReportGenerator(output_dir="demo_reports")

    # Load sample data
    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data = {}
    for symbol in symbols:
        try:
            stock_data = data_loader.download_ohlcv(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            data[symbol] = stock_data
        except Exception:
            pass

    if not data:
        data = create_synthetic_data()

    # Create simple backtest
    sample_data = next(iter(data.values()))

    # Generate simple momentum signals
    signals = generate_momentum_signals(sample_data)

    # Run backtest with Astra VectorBT wrapper
    config = AstraBacktestConfig(initial_cash=100000, commission=0.001)
    backtest_engine = create_astra_backtest(config)

    try:
        result = backtest_engine.run_backtest(sample_data, signals)
    except Exception:
        result = create_mock_backtest_result()

    # Standard performance charts
    try:
        fig1 = chart_builder.plot_backtest_results(result, save_path="demo_reports/performance_chart.png")
        plt.close(fig1)
    except Exception:
        pass

    # VectorBT-inspired interactive charts
    try:
        fig2 = vectorbt_charts.create_interactive_backtest_chart(
            portfolio_value=result.portfolio_value,
            trades=result.trades,
        )
        fig2.write_html("demo_reports/interactive_backtest.html")
    except Exception:
        pass

    # Run Monte Carlo scenarios
    try:
        flash_crash = FlashCrashScenario()
        mc_result = flash_crash.run([100.0], num_simulations=1000, time_steps=252)

        fig3 = vectorbt_charts.create_monte_carlo_visualization(
            paths=mc_result.paths,
            title="Flash Crash Scenario Analysis",
        )
        fig3.write_html("demo_reports/monte_carlo_flash_crash.html")
    except Exception:
        pass

    # Create risk monitoring data
    try:
        risk_data = create_sample_risk_data(result)

        fig4 = chart_builder.plot_risk_dashboard(risk_data, save_path="demo_reports/risk_dashboard.png")
        plt.close(fig4)
    except Exception:
        pass

    # Generate comprehensive reports
    with contextlib.suppress(Exception):
        report_path = report_generator.generate_backtest_report(
            result,
            strategy_name="Momentum Demo",
            benchmark_data=sample_data["Close"],
        )

    # Strategy comparison
    try:
        strategy_results = {
            "Momentum": {
                "Total Return": result.metrics.get("total_return", 0),
                "Sharpe Ratio": result.metrics.get("sharpe_ratio", 0),
                "Max Drawdown": abs(result.metrics.get("max_drawdown", 0)),
                "Win Rate": result.metrics.get("win_rate", 0.5),
                "Profit Factor": result.metrics.get("profit_factor", 1.0),
            },
            "Buy & Hold": {
                "Total Return": 0.12,
                "Sharpe Ratio": 0.8,
                "Max Drawdown": 0.15,
                "Win Rate": 0.6,
                "Profit Factor": 1.2,
            },
        }

        fig5 = vectorbt_charts.create_strategy_comparison(strategy_results)
        fig5.write_html("demo_reports/strategy_comparison.html")
    except Exception:
        pass

    # Correlation matrix
    try:
        if len(data) > 1:
            returns_data = pd.DataFrame({
                symbol: stock_data["Close"].pct_change().fillna(0)
                for symbol, stock_data in data.items()
            })

            fig6 = vectorbt_charts.create_correlation_matrix(returns_data)
            fig6.write_html("demo_reports/correlation_matrix.html")
    except Exception:
        pass

    # Create comprehensive dashboard
    try:
        dashboard_data = {
            "portfolio_values": result.portfolio_value,
            "risk_score": 45,
            "positions": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
            "monte_carlo_paths": mc_result.paths[:10] if "mc_result" in locals() else [],
        }

        fig7 = vectorbt_charts.create_interactive_dashboard(dashboard_data)
        fig7.write_html("demo_reports/interactive_dashboard.html")
    except Exception:
        pass


def create_synthetic_data() -> dict:
    """Create synthetic data for demo purposes."""
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    # Generate realistic stock data with trends and volatility
    np.random.seed(42)

    data = {}
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
        prices = 100 * np.cumprod(1 + returns)

        # Add some realistic OHLC data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
        volumes = np.random.randint(1000000, 10000000, len(dates))

        data[symbol] = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        }, index=dates)

    return data

def generate_momentum_signals(data: pd.DataFrame) -> pd.Series:
    """Generate simple momentum signals."""
    # Simple moving average crossover
    short_ma = data["Close"].rolling(20).mean()
    long_ma = data["Close"].rolling(50).mean()

    signals = pd.Series(0, index=data.index)
    signals[short_ma > long_ma] = 1
    signals[short_ma < long_ma] = -1

    return signals

def create_mock_backtest_result():
    """Create mock backtest result for demo."""
    from src.backtesting.engine import BacktestResult

    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    # Generate mock portfolio values
    returns = np.random.normal(0.0005, 0.015, len(dates))
    portfolio_values = 100000 * np.cumprod(1 + returns)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    returns_series = pd.Series(returns, index=dates)

    trades_df = pd.DataFrame({
        "symbol": ["DEMO"] * 10,
        "quantity": np.random.randint(-100, 100, 10),
        "price": np.random.uniform(90, 110, 10),
        "value": np.random.uniform(9000, 11000, 10),
    })

    metrics = {
        "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1,
        "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
        "max_drawdown": -0.08,
        "calmar_ratio": 1.2,
        "win_rate": 0.55,
        "profit_factor": 1.3,
    }

    return BacktestResult(
        portfolio_value=portfolio_series,
        returns=returns_series,
        trades=trades_df,
        metrics=metrics,
    )

def create_sample_risk_data(backtest_result):
    """Create sample risk monitoring data."""
    return {
        "current_metrics": type("obj", (object,), {
            "volatility": 0.18,
            "max_drawdown": -0.08,
            "var_95": -0.03,
        })(),
        "value_history": backtest_result.portfolio_value,
        "positions": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        "risk_score_history": pd.Series(
            np.random.uniform(30, 70, len(backtest_result.portfolio_value)),
            index=backtest_result.portfolio_value.index,
        ),
        "recent_alerts": [],
        "circuit_breaker_status": {
            "drawdown_breaker": {"enabled": True, "breach_count": 0},
            "volatility_breaker": {"enabled": True, "breach_count": 1},
            "var_breaker": {"enabled": True, "breach_count": 0},
        },
    }

if __name__ == "__main__":
    main()
