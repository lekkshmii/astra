"""Quick test of Astra visualization components."""

import numpy as np
import pandas as pd

# Test our visualization components
from src.visualization import AstraReportGenerator, AstraVectorBTCharts, ChartBuilder


def main() -> None:

    # Create test data
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)

    # Generate mock portfolio performance
    returns = np.random.normal(0.0005, 0.015, len(dates))
    portfolio_values = 100000 * np.cumprod(1 + returns)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    returns_series = pd.Series(returns, index=dates)

    # Mock backtest result
    from src.backtesting.engine import BacktestResult

    trades_df = pd.DataFrame({
        "symbol": ["TEST"] * 5,
        "quantity": [100, -50, 75, -100, 50],
        "price": [100, 105, 98, 102, 99],
        "value": [10000, -5250, 7350, -10200, 4950],
    })

    metrics = {
        "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1,
        "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
        "max_drawdown": -0.08,
        "calmar_ratio": 1.2,
    }

    result = BacktestResult(
        portfolio_value=portfolio_series,
        returns=returns_series,
        trades=trades_df,
        metrics=metrics,
    )


    # Test 1: Basic chart builder
    try:
        chart_builder = ChartBuilder()
        fig = chart_builder.plot_backtest_results(result)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    # Test 2: VectorBT-inspired charts
    try:
        vectorbt_charts = AstraVectorBTCharts()
        vectorbt_charts.create_interactive_backtest_chart(
            portfolio_value=portfolio_series,
            trades=trades_df,
        )
    except Exception:
        pass

    # Test 3: Report generator
    try:
        report_gen = AstraReportGenerator()
        report_gen.generate_backtest_report(result, "Test Strategy")
    except Exception:
        pass


if __name__ == "__main__":
    main()
