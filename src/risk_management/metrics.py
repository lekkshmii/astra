"""Risk Metrics Calculator - Astra Trading Platform.
================================================

Comprehensive risk metrics calculation for portfolio analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class RiskMetrics:
    """Comprehensive risk metrics calculator.

    Provides institutional-grade risk measurement capabilities.
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize risk metrics calculator.

        Args:
        ----
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation

        """
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(self,
                                      returns: pd.Series,
                                      portfolio_value: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive risk and performance metrics.

        Args:
        ----
            returns: Daily returns series
            portfolio_value: Optional portfolio value series

        Returns:
        -------
            Dictionary of calculated metrics

        """
        metrics = {}

        # Basic return metrics
        metrics["total_return"] = self._calculate_total_return(returns)
        metrics["annualized_return"] = self._calculate_annualized_return(returns)

        # Risk metrics
        metrics["volatility"] = self._calculate_volatility(returns)
        metrics["max_drawdown"] = self._calculate_max_drawdown(returns)
        metrics["avg_drawdown"] = self._calculate_avg_drawdown(returns)

        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(returns)
        metrics["sortino_ratio"] = self._calculate_sortino_ratio(returns)

        # VaR and tail risk
        metrics["var_95"] = self._calculate_var(returns, 0.95)
        metrics["var_99"] = self._calculate_var(returns, 0.99)
        metrics["expected_shortfall"] = self._calculate_expected_shortfall(returns, 0.95)

        # Distribution metrics
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()

        # Performance statistics
        metrics["best_day"] = returns.max()
        metrics["worst_day"] = returns.min()

        return metrics

    def calculate_risk_score(self,
                           returns: pd.Series,
                           portfolio_value: Optional[pd.Series] = None) -> float:
        """Calculate overall risk score (0-100 scale).

        Args:
        ----
            returns: Daily returns series
            portfolio_value: Optional portfolio value series

        Returns:
        -------
            Risk score between 0 (low risk) and 100 (high risk)

        """
        # Calculate component scores
        vol_score = min(100, (returns.std() * np.sqrt(252)) / 0.30 * 100)  # Scale to 30% max
        dd_score = min(100, abs(self._calculate_max_drawdown(returns)) / 0.20 * 100)  # Scale to 20% max
        var_score = min(100, abs(self._calculate_var(returns, 0.95)) / 0.05 * 100)  # Scale to 5% max

        # Weighted average
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + var_score * 0.2)

        return min(100, max(0, risk_score))

    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        return (1 + returns).prod() - 1

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        years = len(returns) / 252
        total_return = self._calculate_total_return(returns)
        return (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _calculate_avg_drawdown(self, returns: pd.Series) -> float:
        """Calculate average drawdown."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown[drawdown < 0].mean()

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - target_return / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)

        return (excess_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_rolling_metrics(self,
                                returns: pd.Series,
                                window: int = 252) -> pd.DataFrame:
        """Calculate rolling risk metrics.

        Args:
        ----
            returns: Daily returns series
            window: Rolling window size (default 252 = 1 year)

        Returns:
        -------
            DataFrame with rolling metrics

        """
        rolling_metrics = pd.DataFrame(index=returns.index)

        rolling_metrics["volatility"] = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics["sharpe_ratio"] = self._rolling_sharpe(returns, window)
        rolling_metrics["max_drawdown"] = self._rolling_max_drawdown(returns, window)
        rolling_metrics["var_95"] = returns.rolling(window).quantile(0.05)

        return rolling_metrics

    def _rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return (excess_returns.rolling(window).mean() /
                excess_returns.rolling(window).std() * np.sqrt(252))

    def _rolling_max_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        def max_dd(x):
            cumulative = (1 + x).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            return drawdown.min()

        return returns.rolling(window).apply(max_dd, raw=False)

    def compare_metrics(self,
                       returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare metrics across multiple return series.

        Args:
        ----
            returns_dict: Dictionary of return series

        Returns:
        -------
            DataFrame with comparison metrics

        """
        comparison_data = []

        for name, returns in returns_dict.items():
            metrics = self.calculate_comprehensive_metrics(returns)
            metrics["Strategy"] = name
            comparison_data.append(metrics)

        return pd.DataFrame(comparison_data)

    def risk_attribution(self,
                        portfolio_returns: pd.Series,
                        asset_returns: Dict[str, pd.Series],
                        weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate risk attribution across portfolio components.

        Args:
        ----
            portfolio_returns: Portfolio total returns
            asset_returns: Individual asset returns
            weights: Asset weights in portfolio

        Returns:
        -------
            Dictionary with risk attribution results

        """
        attribution = {}

        portfolio_var = portfolio_returns.var()

        for asset, returns in asset_returns.items():
            if asset in weights:
                weight = weights[asset]
                asset_var = returns.var()

                # Calculate contribution to portfolio variance
                covariance_sum = sum(
                    weights.get(other_asset, 0) * returns.cov(other_returns)
                    for other_asset, other_returns in asset_returns.items()
                    if other_asset in weights
                )

                var_contribution = weight * covariance_sum / portfolio_var if portfolio_var > 0 else 0

                attribution[asset] = {
                    "weight": weight,
                    "volatility": np.sqrt(asset_var * 252),
                    "var_contribution": var_contribution,
                    "risk_contribution": var_contribution * 100,
                }

        return attribution
