"""Astra VectorBT Wrapper - Professional Backtesting Interface.
===========================================================

Professional wrapper around VectorBT with Astra branding and enhanced features.
Falls back to custom Astra engine when VectorBT is unavailable.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Using Astra custom backtesting engine.")

from .engine import BacktestEngine, BacktestResult


@dataclass
class AstraBacktestConfig:
    """Astra backtesting configuration."""

    initial_cash: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    freq: str = "D"
    compound: bool = True

class AstraVectorBT:
    """Professional VectorBT wrapper with Astra branding.

    Provides a unified interface for backtesting with enhanced features:
    - Automatic fallback to custom Astra engine
    - Professional result formatting
    - Enhanced performance metrics
    - Risk-adjusted returns
    """

    def __init__(self, config: Optional[AstraBacktestConfig] = None) -> None:
        """Initialize Astra VectorBT wrapper."""
        self.config = config or AstraBacktestConfig()
        self.use_vectorbt = VECTORBT_AVAILABLE

        if not self.use_vectorbt:
            # Fallback to custom Astra engine
            self.engine = BacktestEngine(
                initial_capital=self.config.initial_cash,
                commission=self.config.commission,
            )

    def run_backtest(self,
                    data: pd.DataFrame,
                    signals: Union[pd.DataFrame, pd.Series],
                    **kwargs) -> "AstraBacktestResult":
        """Run backtest with automatic engine selection.

        Args:
        ----
            data: Price data (OHLCV format)
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            **kwargs: Additional parameters

        Returns:
        -------
            AstraBacktestResult: Professional backtest results

        """
        if self.use_vectorbt:
            return self._run_vectorbt_backtest(data, signals, **kwargs)
        else:
            return self._run_astra_backtest(data, signals, **kwargs)

    def _run_vectorbt_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, **kwargs) -> "AstraBacktestResult":
        """Run backtest using VectorBT engine."""
        try:
            # Configure VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                data["Close"] if "Close" in data.columns else data,
                signals,
                init_cash=self.config.initial_cash,
                fees=self.config.commission,
                slippage=self.config.slippage,
                freq=self.config.freq,
                **kwargs,
            )

            # Extract results
            returns = portfolio.returns()
            trades = portfolio.trades.records_readable

            # Calculate enhanced metrics
            metrics = self._calculate_enhanced_metrics(portfolio)

            return AstraBacktestResult(
                portfolio_value=portfolio.value(),
                returns=returns,
                trades=trades,
                metrics=metrics,
                engine="VectorBT",
                portfolio=portfolio,
            )

        except Exception as e:
            warnings.warn(f"VectorBT backtest failed: {e}. Falling back to Astra engine.")
            return self._run_astra_backtest(data, signals, **kwargs)

    def _run_astra_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, **kwargs) -> "AstraBacktestResult":
        """Run backtest using custom Astra engine."""
        # Convert signals to format expected by Astra engine
        if isinstance(signals, pd.Series):
            signals_df = pd.DataFrame({"signal": signals})
        else:
            signals_df = signals

        # Run backtest
        result = self.engine.run_backtest(data, signals_df)

        # Enhanced metrics calculation
        enhanced_metrics = self._calculate_astra_metrics(result)

        return AstraBacktestResult(
            portfolio_value=result.portfolio_value,
            returns=result.returns,
            trades=result.trades,
            metrics={**result.metrics, **enhanced_metrics},
            engine="Astra Custom",
            portfolio=None,
        )

    def _calculate_enhanced_metrics(self, portfolio) -> Dict[str, float]:
        """Calculate enhanced metrics for VectorBT results."""
        metrics = {}

        try:
            stats = portfolio.stats()
            metrics.update({
                "total_return": stats["Total Return [%]"] / 100,
                "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                "max_drawdown": stats.get("Max Drawdown [%]", 0) / 100,
                "calmar_ratio": stats.get("Calmar Ratio", 0),
                "win_rate": stats.get("Win Rate [%]", 0) / 100,
                "profit_factor": stats.get("Profit Factor", 0),
                "volatility": portfolio.returns().std() * np.sqrt(252),
                "sortino_ratio": self._calculate_sortino(portfolio.returns()),
            })
        except Exception as e:
            warnings.warn(f"Could not calculate enhanced metrics: {e}")

        return metrics

    def _calculate_astra_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate enhanced metrics for Astra engine results."""
        returns = result.returns

        return {
            "volatility": returns.std() * np.sqrt(252),
            "sortino_ratio": self._calculate_sortino(returns),
            "win_rate": (returns > 0).mean(),
            "profit_factor": self._calculate_profit_factor(returns),
            "var_95": np.percentile(returns, 5),
            "var_99": np.percentile(returns, 1),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }


    def _calculate_sortino(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - target_return / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)

        if downside_deviation == 0:
            return 0

        return (excess_returns.mean() * 252) / downside_deviation

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 0

        return gains / losses

    def create_portfolio_comparison(self,
                                  strategies: Dict[str, pd.DataFrame],
                                  data: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple strategies."""
        results = {}

        for name, signals in strategies.items():
            try:
                result = self.run_backtest(data, signals)
                results[name] = {
                    "Total Return": result.metrics.get("total_return", 0),
                    "Sharpe Ratio": result.metrics.get("sharpe_ratio", 0),
                    "Max Drawdown": result.metrics.get("max_drawdown", 0),
                    "Win Rate": result.metrics.get("win_rate", 0),
                    "Volatility": result.metrics.get("volatility", 0),
                    "Sortino Ratio": result.metrics.get("sortino_ratio", 0),
                }
            except Exception as e:
                warnings.warn(f"Strategy {name} failed: {e}")

        return pd.DataFrame(results).T


@dataclass
class AstraBacktestResult:
    """Enhanced backtest results with Astra branding."""

    portfolio_value: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    engine: str
    portfolio: Optional[Any] = None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get formatted performance summary."""
        return {
            "Engine": self.engine,
            "Total Return": f"{self.metrics.get('total_return', 0):.2%}",
            "Sharpe Ratio": f"{self.metrics.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{self.metrics.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{self.metrics.get('win_rate', 0):.2%}",
            "Volatility": f"{self.metrics.get('volatility', 0):.2%}",
            "Sortino Ratio": f"{self.metrics.get('sortino_ratio', 0):.2f}",
            "Profit Factor": f"{self.metrics.get('profit_factor', 0):.2f}",
        }

    def plot_performance(self, figsize: tuple = (12, 8)) -> None:
        """Plot performance charts."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle("Astra Backtest Performance Analysis", fontsize=16, fontweight="bold")

            # Portfolio value
            axes[0, 0].plot(self.portfolio_value.index, self.portfolio_value.values)
            axes[0, 0].set_title("Portfolio Value")
            axes[0, 0].grid(True)

            # Returns distribution
            axes[0, 1].hist(self.returns.dropna(), bins=50, alpha=0.7)
            axes[0, 1].set_title("Returns Distribution")
            axes[0, 1].grid(True)

            # Cumulative returns
            cumulative = (1 + self.returns).cumprod()
            axes[1, 0].plot(cumulative.index, cumulative.values)
            axes[1, 0].set_title("Cumulative Returns")
            axes[1, 0].grid(True)

            # Drawdown
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color="red")
            axes[1, 1].set_title("Drawdown")
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show()

        except Exception:
            pass


def create_astra_backtest(config: Optional[AstraBacktestConfig] = None) -> AstraVectorBT:
    """Factory function to create Astra backtesting engine.

    Args:
    ----
        config: Optional configuration

    Returns:
    -------
        AstraVectorBT: Configured backtesting engine

    """
    return AstraVectorBT(config)
