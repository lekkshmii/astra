"""Position Sizing - Astra Trading Platform.
========================================

Dynamic position sizing with Monte Carlo risk inputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation."""

    symbol: str
    target_weight: float
    target_shares: float
    target_value: float
    risk_contribution: float
    confidence_level: float

class BasePositionSizer(ABC):
    """Base class for position sizing methods."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def calculate_positions(self,
                          portfolio_value: float,
                          signals: pd.Series,
                          prices: pd.Series,
                          **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate position sizes."""

class FixedFractionalSizer(BasePositionSizer):
    """Fixed fractional position sizing.

    Allocates fixed percentage of capital per position.
    """

    def __init__(self, fraction_per_position: float = 0.20) -> None:
        super().__init__("FixedFractional")
        self.fraction_per_position = fraction_per_position

    def calculate_positions(self,
                          portfolio_value: float,
                          signals: pd.Series,
                          prices: pd.Series,
                          **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate fixed fractional positions."""
        positions = {}
        active_signals = signals[signals != 0]

        if len(active_signals) == 0:
            return positions

        allocation_per_position = self.fraction_per_position

        for symbol, signal in active_signals.items():
            if symbol in prices and prices[symbol] > 0:
                target_value = portfolio_value * allocation_per_position * signal
                target_shares = target_value / prices[symbol]
                target_weight = allocation_per_position * signal

                positions[symbol] = PositionSizeResult(
                    symbol=symbol,
                    target_weight=target_weight,
                    target_shares=target_shares,
                    target_value=target_value,
                    risk_contribution=abs(target_weight),
                    confidence_level=0.95,  # Default
                )

        return positions

class VolatilityTargetSizer(BasePositionSizer):
    """Volatility targeting position sizer.

    Sizes positions to target specific volatility contribution.
    """

    def __init__(self,
                 target_volatility: float = 0.15,  # 15% target vol
                 lookback_window: int = 20) -> None:
        super().__init__("VolatilityTarget")
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window

    def calculate_positions(self,
                          portfolio_value: float,
                          signals: pd.Series,
                          prices: pd.Series,
                          returns_data: Optional[pd.DataFrame] = None,
                          **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate volatility-targeted positions."""
        positions = {}
        active_signals = signals[signals != 0]

        if len(active_signals) == 0 or returns_data is None:
            return positions

        # Calculate recent volatilities
        recent_returns = returns_data.tail(self.lookback_window)
        volatilities = recent_returns.std() * np.sqrt(252)  # Annualized

        for symbol, signal in active_signals.items():
            if symbol in prices and symbol in volatilities and volatilities[symbol] > 0:
                # Calculate position size based on volatility
                asset_vol = volatilities[symbol]
                vol_scalar = self.target_volatility / asset_vol
                base_weight = 1.0 / len(active_signals)  # Equal weight base
                adjusted_weight = base_weight * vol_scalar * signal

                # Cap maximum position size
                adjusted_weight = np.clip(adjusted_weight, -0.25, 0.25)

                target_value = portfolio_value * adjusted_weight
                target_shares = target_value / prices[symbol]

                positions[symbol] = PositionSizeResult(
                    symbol=symbol,
                    target_weight=adjusted_weight,
                    target_shares=target_shares,
                    target_value=target_value,
                    risk_contribution=abs(adjusted_weight) * asset_vol / self.target_volatility,
                    confidence_level=0.90,
                )

        return positions

class KellyCriterionSizer(BasePositionSizer):
    """Kelly Criterion position sizer.

    Optimizes position size based on expected returns and win probability.
    """

    def __init__(self,
                 max_kelly_fraction: float = 0.25,  # Cap Kelly at 25%
                 lookback_window: int = 50) -> None:
        super().__init__("KellyCriterion")
        self.max_kelly_fraction = max_kelly_fraction
        self.lookback_window = lookback_window

    def calculate_positions(self,
                          portfolio_value: float,
                          signals: pd.Series,
                          prices: pd.Series,
                          returns_data: Optional[pd.DataFrame] = None,
                          **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate Kelly Criterion positions."""
        positions = {}
        active_signals = signals[signals != 0]

        if len(active_signals) == 0 or returns_data is None:
            return positions

        recent_returns = returns_data.tail(self.lookback_window)

        for symbol, signal in active_signals.items():
            if symbol in prices and symbol in recent_returns.columns:
                asset_returns = recent_returns[symbol].dropna()

                if len(asset_returns) < 20:  # Minimum data requirement
                    continue

                # Calculate Kelly fraction
                asset_returns.mean()
                win_prob = (asset_returns > 0).mean()

                if win_prob > 0 and win_prob < 1:
                    avg_win = asset_returns[asset_returns > 0].mean()
                    avg_loss = abs(asset_returns[asset_returns < 0].mean())

                    if avg_loss > 0:
                        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                        kelly_fraction = np.clip(kelly_fraction, 0, self.max_kelly_fraction)
                        kelly_fraction *= signal  # Apply signal direction

                        target_value = portfolio_value * kelly_fraction
                        target_shares = target_value / prices[symbol]

                        positions[symbol] = PositionSizeResult(
                            symbol=symbol,
                            target_weight=kelly_fraction,
                            target_shares=target_shares,
                            target_value=target_value,
                            risk_contribution=abs(kelly_fraction),
                            confidence_level=win_prob,
                        )

        return positions

class MonteCarloSizer(BasePositionSizer):
    """Monte Carlo informed position sizer.

    Uses Monte Carlo scenario results to size positions.
    """

    def __init__(self,
                 target_var_95: float = 0.05,  # 5% VaR target
                 confidence_threshold: float = 0.90) -> None:
        super().__init__("MonteCarlo")
        self.target_var_95 = target_var_95
        self.confidence_threshold = confidence_threshold

    def calculate_positions(self,
                          portfolio_value: float,
                          signals: pd.Series,
                          prices: pd.Series,
                          monte_carlo_results: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate Monte Carlo informed positions."""
        positions = {}
        active_signals = signals[signals != 0]

        if len(active_signals) == 0:
            return positions

        if monte_carlo_results is None:
            # Fallback to equal weight
            return self._equal_weight_fallback(portfolio_value, active_signals, prices)

        # Extract VaR estimates from Monte Carlo results
        var_estimates = monte_carlo_results.get("var_estimates", {})

        total_risk_budget = self.target_var_95

        for symbol, signal in active_signals.items():
            if symbol in prices and prices[symbol] > 0:
                # Get VaR estimate for this asset
                asset_var = var_estimates.get(symbol, 0.15)  # Default 15% VaR

                # Calculate position size to stay within risk budget
                if asset_var > 0:
                    max_weight_for_var = total_risk_budget / asset_var / len(active_signals)
                    target_weight = min(max_weight_for_var, 0.20) * signal  # Cap at 20%

                    target_value = portfolio_value * target_weight
                    target_shares = target_value / prices[symbol]

                    # Calculate confidence based on Monte Carlo success rate
                    success_rate = monte_carlo_results.get("success_rates", {}).get(symbol, 0.5)
                    confidence = max(success_rate, 0.5)  # Minimum 50% confidence

                    positions[symbol] = PositionSizeResult(
                        symbol=symbol,
                        target_weight=target_weight,
                        target_shares=target_shares,
                        target_value=target_value,
                        risk_contribution=abs(target_weight) * asset_var,
                        confidence_level=confidence,
                    )

        return positions

    def _equal_weight_fallback(self,
                              portfolio_value: float,
                              active_signals: pd.Series,
                              prices: pd.Series) -> Dict[str, PositionSizeResult]:
        """Fallback to equal weight when no Monte Carlo data."""
        positions = {}
        equal_weight = 0.20 / len(active_signals)  # 20% total exposure

        for symbol, signal in active_signals.items():
            if symbol in prices and prices[symbol] > 0:
                target_weight = equal_weight * signal
                target_value = portfolio_value * target_weight
                target_shares = target_value / prices[symbol]

                positions[symbol] = PositionSizeResult(
                    symbol=symbol,
                    target_weight=target_weight,
                    target_shares=target_shares,
                    target_value=target_value,
                    risk_contribution=abs(target_weight) * 0.15,  # Assume 15% vol
                    confidence_level=0.70,
                )

        return positions

class PositionSizer:
    """Position sizing coordinator.

    Manages multiple position sizing methods and provides unified interface.
    """

    def __init__(self, default_method: str = "VolatilityTarget") -> None:
        """Initialize position sizer."""
        self.methods = {
            "FixedFractional": FixedFractionalSizer(),
            "VolatilityTarget": VolatilityTargetSizer(),
            "KellyCriterion": KellyCriterionSizer(),
            "MonteCarlo": MonteCarloSizer(),
        }
        self.default_method = default_method

    def size_positions(self,
                      portfolio_value: float,
                      signals: pd.Series,
                      prices: pd.Series,
                      method: Optional[str] = None,
                      **kwargs) -> Dict[str, PositionSizeResult]:
        """Calculate position sizes using specified method.

        Args:
        ----
            portfolio_value: Total portfolio value
            signals: Trading signals (-1, 0, 1)
            prices: Current asset prices
            method: Position sizing method name
            **kwargs: Additional data for specific methods

        Returns:
        -------
            Dictionary of position sizing results

        """
        method_name = method or self.default_method

        if method_name not in self.methods:
            msg = f"Unknown position sizing method: {method_name}"
            raise ValueError(msg)

        sizer = self.methods[method_name]
        return sizer.calculate_positions(portfolio_value, signals, prices, **kwargs)

    def get_portfolio_summary(self, positions: Dict[str, PositionSizeResult]) -> Dict[str, float]:
        """Get portfolio-level summary from positions."""
        if not positions:
            return {}

        total_weight = sum(pos.target_weight for pos in positions.values())
        total_risk = sum(pos.risk_contribution for pos in positions.values())
        avg_confidence = np.mean([pos.confidence_level for pos in positions.values()])

        return {
            "total_weight": total_weight,
            "total_risk_contribution": total_risk,
            "average_confidence": avg_confidence,
            "number_of_positions": len(positions),
            "largest_position": max(abs(pos.target_weight) for pos in positions.values()),
            "position_concentration": max(abs(pos.target_weight) for pos in positions.values()) / max(total_weight, 0.01),
        }

    def validate_positions(self,
                          positions: Dict[str, PositionSizeResult],
                          max_total_weight: float = 1.0,
                          max_single_position: float = 0.30) -> Tuple[bool, List[str]]:
        """Validate position sizing results.

        Args:
        ----
            positions: Position sizing results
            max_total_weight: Maximum total portfolio weight
            max_single_position: Maximum single position weight

        Returns:
        -------
            Tuple of (is_valid, list_of_issues)

        """
        issues = []

        if not positions:
            return True, issues

        total_weight = sum(abs(pos.target_weight) for pos in positions.values())
        if total_weight > max_total_weight:
            issues.append(f"Total weight {total_weight:.1%} exceeds maximum {max_total_weight:.1%}")

        for symbol, pos in positions.items():
            if abs(pos.target_weight) > max_single_position:
                issues.append(f"Position {symbol} weight {pos.target_weight:.1%} exceeds maximum {max_single_position:.1%}")

        return len(issues) == 0, issues

    # Simple methods for notebook compatibility
    def fixed_fractional(self, portfolio_value: float, fraction: float = 0.05) -> Dict[str, float]:
        """Simple fixed fractional position sizing for notebooks."""
        return {
            "position_size": portfolio_value * fraction,
            "fraction": fraction,
            "method": "fixed_fractional",
        }

    def volatility_target(self, portfolio_value: float, target_vol: float = 0.15) -> Dict[str, float]:
        """Simple volatility targeting for notebooks."""
        return {
            "position_size": portfolio_value * 0.20,  # Default allocation
            "target_volatility": target_vol,
            "method": "volatility_target",
        }

    def kelly_criterion(self, win_rate: float = 0.55, avg_win: float = 0.02, avg_loss: float = 0.015) -> float:
        """Simple Kelly criterion calculation for notebooks."""
        if avg_win > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0, min(kelly, 0.25))  # Cap at 25%
        return 0.05  # Default 5%

    def monte_carlo_informed(self, portfolio_value: float, var_estimate: float = 0.05) -> Dict[str, float]:
        """Simple Monte Carlo informed sizing for notebooks."""
        target_allocation = min(0.20, 0.05 / var_estimate)  # 5% VaR target
        return {
            "position_size": portfolio_value * target_allocation,
            "var_estimate": var_estimate,
            "allocation": target_allocation,
            "method": "monte_carlo_informed",
        }
