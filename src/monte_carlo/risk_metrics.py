"""Risk Metrics Calculator - Astra Trading Platform.
================================================

Professional risk metrics calculation from Monte Carlo results.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


class RiskCalculator:
    """Professional risk metrics calculator.

    Calculates institutional-grade risk measures from Monte Carlo simulations.
    """

    def __init__(self) -> None:
        """Initialize risk calculator."""

    def calculate_var(self,
                     returns: np.ndarray,
                     confidence_levels: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) at multiple confidence levels.

        Args:
        ----
            returns: Array of portfolio returns
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])

        Returns:
        -------
            Dictionary of VaR values by confidence level

        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        var_results = {}

        for confidence in confidence_levels:
            var_level = (1 - confidence) * 100
            var_value = np.percentile(returns, var_level)
            var_results[f"VaR_{int(confidence*100)}"] = var_value

        return var_results

    def calculate_expected_shortfall(self,
                                   returns: np.ndarray,
                                   confidence_levels: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate Expected Shortfall (Conditional VaR).

        Args:
        ----
            returns: Array of portfolio returns
            confidence_levels: List of confidence levels

        Returns:
        -------
            Dictionary of Expected Shortfall values

        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        es_results = {}

        for confidence in confidence_levels:
            var_level = (1 - confidence) * 100
            var_threshold = np.percentile(returns, var_level)
            tail_losses = returns[returns <= var_threshold]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
            es_results[f"ES_{int(confidence*100)}"] = expected_shortfall

        return es_results

    def calculate_drawdown_metrics(self, price_paths: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics.

        Args:
        ----
            price_paths: Array of price paths (simulations x time_steps)

        Returns:
        -------
            Dictionary of drawdown metrics

        """
        max_drawdowns = []
        drawdown_durations = []
        time_to_recovery = []

        for path in price_paths:
            # Calculate running maximum (peak)
            peak = np.maximum.accumulate(path)

            # Calculate drawdown
            drawdown = (path - peak) / peak
            max_dd = np.min(drawdown)
            max_drawdowns.append(max_dd)

            # Calculate drawdown duration
            in_drawdown = drawdown < -0.01  # 1% threshold
            if np.any(in_drawdown):
                drawdown_periods = self._get_consecutive_periods(in_drawdown)
                if drawdown_periods:
                    drawdown_durations.append(np.max(drawdown_periods))

                    # Time to recovery (simplified)
                    recovery_times = []
                    for i in range(len(path)):
                        if drawdown[i] < -0.05:  # 5% drawdown
                            # Look for recovery
                            for j in range(i, len(path)):
                                if path[j] >= peak[i]:
                                    recovery_times.append(j - i)
                                    break
                    if recovery_times:
                        time_to_recovery.append(np.mean(recovery_times))

        return {
            "max_drawdown_mean": np.mean(max_drawdowns),
            "max_drawdown_std": np.std(max_drawdowns),
            "max_drawdown_worst": np.min(max_drawdowns),
            "max_drawdown_5th_percentile": np.percentile(max_drawdowns, 5),
            "drawdown_duration_mean": np.mean(drawdown_durations) if drawdown_durations else 0,
            "time_to_recovery_mean": np.mean(time_to_recovery) if time_to_recovery else 0,
        }

    def calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate tail risk metrics.

        Args:
        ----
            returns: Array of returns

        Returns:
        -------
            Dictionary of tail risk metrics

        """
        # Sort returns for tail analysis
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)

        # Tail metrics
        left_tail_5 = sorted_returns[:int(0.05 * n)]
        left_tail_1 = sorted_returns[:int(0.01 * n)]

        return {
            "skewness": stats.skew(returns),
            "kurtosis": stats.kurtosis(returns),
            "left_tail_mean_5pct": np.mean(left_tail_5) if len(left_tail_5) > 0 else 0,
            "left_tail_mean_1pct": np.mean(left_tail_1) if len(left_tail_1) > 0 else 0,
            "tail_ratio": abs(np.percentile(returns, 5)) / np.percentile(returns, 95),
            "downside_deviation": self._calculate_downside_deviation(returns),
        }

    def calculate_stress_test_metrics(self,
                                    baseline_returns: np.ndarray,
                                    stress_scenario_results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate stress test metrics comparing baseline to stress scenarios.

        Args:
        ----
            baseline_returns: Baseline scenario returns
            stress_scenario_results: Dict of scenario_name -> returns

        Returns:
        -------
            Stress test comparison metrics

        """
        baseline_metrics = self._calculate_basic_metrics(baseline_returns)

        stress_metrics = {}

        for scenario_name, scenario_returns in stress_scenario_results.items():
            scenario_metrics = self._calculate_basic_metrics(scenario_returns)

            # Calculate deltas from baseline
            stress_metrics[scenario_name] = {
                "return_delta": scenario_metrics["mean_return"] - baseline_metrics["mean_return"],
                "volatility_delta": scenario_metrics["volatility"] - baseline_metrics["volatility"],
                "var_95_delta": scenario_metrics["var_95"] - baseline_metrics["var_95"],
                "max_loss_delta": scenario_metrics["min_return"] - baseline_metrics["min_return"],
                "scenario_probability": self._estimate_scenario_probability(scenario_name),
            }

        return stress_metrics

    def _get_consecutive_periods(self, boolean_array: np.ndarray) -> List[int]:
        """Get lengths of consecutive True periods."""
        periods = []
        current_length = 0

        for val in boolean_array:
            if val:
                current_length += 1
            else:
                if current_length > 0:
                    periods.append(current_length)
                    current_length = 0

        if current_length > 0:
            periods.append(current_length)

        return periods

    def _calculate_downside_deviation(self, returns: np.ndarray, target: float = 0.0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt(np.mean((downside_returns - target) ** 2))

    def _calculate_basic_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic return metrics."""
        return {
            "mean_return": np.mean(returns),
            "volatility": np.std(returns),
            "var_95": np.percentile(returns, 5),
            "var_99": np.percentile(returns, 1),
            "min_return": np.min(returns),
            "max_return": np.max(returns),
        }

    def _estimate_scenario_probability(self, scenario_name: str) -> float:
        """Estimate scenario probability (placeholder)."""
        probabilities = {
            "flash_crash": 0.02,      # 2% annual probability
            "regime_switching": 0.15,  # 15% annual probability
            "correlation_breakdown": 0.10,  # 10% annual probability
            "volatility_clustering": 0.25,   # 25% annual probability
        }
        return probabilities.get(scenario_name, 0.05)
