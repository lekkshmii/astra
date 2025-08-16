"""Risk Management Parameters - Astra Trading Platform.
===================================================

Centralized risk limit definitions and validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class RiskLevel(Enum):
    """Risk level classifications."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class PositionLimits:
    """Position sizing limits."""

    max_position_weight: float = 0.30  # 30% max per position
    max_sector_weight: float = 0.50    # 50% max per sector
    max_total_exposure: float = 1.00   # 100% max total exposure
    min_position_size: float = 0.01    # 1% minimum position
    max_positions: int = 20            # Maximum number of positions

@dataclass
class DrawdownLimits:
    """Drawdown risk limits."""

    max_portfolio_drawdown: float = 0.15    # 15% max portfolio drawdown
    warning_drawdown: float = 0.10          # 10% warning level
    daily_loss_limit: float = 0.05          # 5% max daily loss
    weekly_loss_limit: float = 0.10         # 10% max weekly loss
    monthly_loss_limit: float = 0.15        # 15% max monthly loss

@dataclass
class VolatilityLimits:
    """Volatility risk limits."""

    max_portfolio_volatility: float = 0.25  # 25% max annual volatility
    volatility_spike_threshold: float = 3.0 # 3x normal volatility trigger
    var_95_limit: float = 0.05              # 5% VaR 95% limit
    var_99_limit: float = 0.08              # 8% VaR 99% limit

@dataclass
class CorrelationLimits:
    """Correlation risk limits."""

    max_correlation: float = 0.80           # 80% max pairwise correlation
    correlation_lookback: int = 60          # Days for correlation calculation
    correlation_warning: float = 0.70       # 70% correlation warning

@dataclass
class LeverageLimits:
    """Leverage and margin limits."""

    max_gross_leverage: float = 1.50        # 150% max gross leverage
    max_net_leverage: float = 1.00          # 100% max net leverage
    margin_call_threshold: float = 0.30     # 30% margin call threshold
    forced_liquidation: float = 0.20        # 20% forced liquidation

@dataclass
class LiquidityLimits:
    """Liquidity risk limits."""

    min_daily_volume: float = 1000000       # $1M minimum daily volume
    max_volume_participation: float = 0.10  # 10% max of daily volume
    illiquid_position_limit: float = 0.05   # 5% max in illiquid assets

class RiskLimits:
    """Comprehensive risk limits management.

    Provides risk limit definitions for different risk profiles.
    """

    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE) -> None:
        """Initialize risk limits based on risk level."""
        self.risk_level = risk_level
        self._setup_limits()

    def _setup_limits(self) -> None:
        """Setup risk limits based on risk level."""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self._setup_conservative_limits()
        elif self.risk_level == RiskLevel.MODERATE:
            self._setup_moderate_limits()
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self._setup_aggressive_limits()
        else:
            self._setup_moderate_limits()  # Default to moderate

    def _setup_conservative_limits(self) -> None:
        """Setup conservative risk limits."""
        self.position = PositionLimits(
            max_position_weight=0.20,
            max_sector_weight=0.30,
            max_total_exposure=0.80,
            min_position_size=0.02,
            max_positions=15,
        )

        self.drawdown = DrawdownLimits(
            max_portfolio_drawdown=0.10,
            warning_drawdown=0.05,
            daily_loss_limit=0.03,
            weekly_loss_limit=0.06,
            monthly_loss_limit=0.10,
        )

        self.volatility = VolatilityLimits(
            max_portfolio_volatility=0.15,
            volatility_spike_threshold=2.0,
            var_95_limit=0.03,
            var_99_limit=0.05,
        )

        self.correlation = CorrelationLimits(
            max_correlation=0.70,
            correlation_lookback=90,
            correlation_warning=0.60,
        )

        self.leverage = LeverageLimits(
            max_gross_leverage=1.00,
            max_net_leverage=0.80,
            margin_call_threshold=0.40,
            forced_liquidation=0.30,
        )

        self.liquidity = LiquidityLimits(
            min_daily_volume=5000000,
            max_volume_participation=0.05,
            illiquid_position_limit=0.02,
        )

    def _setup_moderate_limits(self) -> None:
        """Setup moderate risk limits (default)."""
        self.position = PositionLimits(
            max_position_weight=0.30,
            max_sector_weight=0.50,
            max_total_exposure=1.00,
            min_position_size=0.01,
            max_positions=20,
        )

        self.drawdown = DrawdownLimits(
            max_portfolio_drawdown=0.15,
            warning_drawdown=0.10,
            daily_loss_limit=0.05,
            weekly_loss_limit=0.10,
            monthly_loss_limit=0.15,
        )

        self.volatility = VolatilityLimits(
            max_portfolio_volatility=0.25,
            volatility_spike_threshold=3.0,
            var_95_limit=0.05,
            var_99_limit=0.08,
        )

        self.correlation = CorrelationLimits(
            max_correlation=0.80,
            correlation_lookback=60,
            correlation_warning=0.70,
        )

        self.leverage = LeverageLimits(
            max_gross_leverage=1.50,
            max_net_leverage=1.00,
            margin_call_threshold=0.30,
            forced_liquidation=0.20,
        )

        self.liquidity = LiquidityLimits(
            min_daily_volume=1000000,
            max_volume_participation=0.10,
            illiquid_position_limit=0.05,
        )

    def _setup_aggressive_limits(self) -> None:
        """Setup aggressive risk limits."""
        self.position = PositionLimits(
            max_position_weight=0.50,
            max_sector_weight=0.70,
            max_total_exposure=1.50,
            min_position_size=0.005,
            max_positions=30,
        )

        self.drawdown = DrawdownLimits(
            max_portfolio_drawdown=0.25,
            warning_drawdown=0.15,
            daily_loss_limit=0.08,
            weekly_loss_limit=0.15,
            monthly_loss_limit=0.25,
        )

        self.volatility = VolatilityLimits(
            max_portfolio_volatility=0.40,
            volatility_spike_threshold=4.0,
            var_95_limit=0.08,
            var_99_limit=0.12,
        )

        self.correlation = CorrelationLimits(
            max_correlation=0.90,
            correlation_lookback=30,
            correlation_warning=0.80,
        )

        self.leverage = LeverageLimits(
            max_gross_leverage=2.00,
            max_net_leverage=1.50,
            margin_call_threshold=0.20,
            forced_liquidation=0.15,
        )

        self.liquidity = LiquidityLimits(
            min_daily_volume=500000,
            max_volume_participation=0.15,
            illiquid_position_limit=0.10,
        )

    def validate_position_weight(self, weight: float, symbol: str = "") -> bool:
        """Validate if position weight is within limits."""
        return abs(weight) <= self.position.max_position_weight

    def validate_total_exposure(self, total_exposure: float) -> bool:
        """Validate if total exposure is within limits."""
        return total_exposure <= self.position.max_total_exposure

    def validate_drawdown(self, drawdown: float) -> bool:
        """Validate if drawdown is within limits."""
        return abs(drawdown) <= self.drawdown.max_portfolio_drawdown

    def validate_daily_loss(self, daily_return: float) -> bool:
        """Validate if daily loss is within limits."""
        return daily_return >= -self.drawdown.daily_loss_limit

    def validate_volatility(self, volatility: float) -> bool:
        """Validate if volatility is within limits."""
        return volatility <= self.volatility.max_portfolio_volatility

    def validate_correlation(self, correlation: float) -> bool:
        """Validate if correlation is within limits."""
        return abs(correlation) <= self.correlation.max_correlation

    def validate_var(self, var_95: float, var_99: float) -> tuple:
        """Validate VaR metrics."""
        var_95_ok = abs(var_95) <= self.volatility.var_95_limit
        var_99_ok = abs(var_99) <= self.volatility.var_99_limit
        return var_95_ok, var_99_ok

    def get_violation_message(self, metric: str, value: float, limit: float) -> str:
        """Get violation message for a specific metric."""
        messages = {
            "position_weight": f"Position weight {value:.1%} exceeds limit {limit:.1%}",
            "total_exposure": f"Total exposure {value:.1%} exceeds limit {limit:.1%}",
            "drawdown": f"Drawdown {value:.1%} exceeds limit {limit:.1%}",
            "daily_loss": f"Daily loss {value:.1%} exceeds limit {limit:.1%}",
            "volatility": f"Volatility {value:.1%} exceeds limit {limit:.1%}",
            "correlation": f"Correlation {value:.1%} exceeds limit {limit:.1%}",
            "var_95": f"VaR 95% {value:.1%} exceeds limit {limit:.1%}",
            "var_99": f"VaR 99% {value:.1%} exceeds limit {limit:.1%}",
        }
        return messages.get(metric, f"Metric {metric} {value} exceeds limit {limit}")

    def get_risk_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all risk limits."""
        return {
            "position_limits": {
                "max_position_weight": self.position.max_position_weight,
                "max_sector_weight": self.position.max_sector_weight,
                "max_total_exposure": self.position.max_total_exposure,
                "max_positions": self.position.max_positions,
            },
            "drawdown_limits": {
                "max_portfolio_drawdown": self.drawdown.max_portfolio_drawdown,
                "daily_loss_limit": self.drawdown.daily_loss_limit,
                "weekly_loss_limit": self.drawdown.weekly_loss_limit,
                "monthly_loss_limit": self.drawdown.monthly_loss_limit,
            },
            "volatility_limits": {
                "max_portfolio_volatility": self.volatility.max_portfolio_volatility,
                "var_95_limit": self.volatility.var_95_limit,
                "var_99_limit": self.volatility.var_99_limit,
            },
            "correlation_limits": {
                "max_correlation": self.correlation.max_correlation,
                "correlation_warning": self.correlation.correlation_warning,
            },
        }

    def update_limits(self, **kwargs) -> None:
        """Update specific risk limits."""
        for section_name, updates in kwargs.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def export_limits(self) -> Dict[str, any]:
        """Export all limits for external use."""
        return {
            "risk_level": self.risk_level.value,
            "position": self.position.__dict__,
            "drawdown": self.drawdown.__dict__,
            "volatility": self.volatility.__dict__,
            "correlation": self.correlation.__dict__,
            "leverage": self.leverage.__dict__,
            "liquidity": self.liquidity.__dict__,
        }
