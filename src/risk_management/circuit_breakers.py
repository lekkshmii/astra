"""Circuit Breakers - Astra Trading Platform.
=========================================

Institutional-grade safety mechanisms for trading systems.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class BreacherSeverity(Enum):
    """Circuit breaker severity levels."""

    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class BreacherEvent:
    """Circuit breaker event record."""

    timestamp: datetime
    breaker_name: str
    severity: BreacherSeverity
    value: float
    threshold: float
    message: str
    action_taken: str

class BaseCircuitBreaker:
    """Base class for all circuit breakers."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.breach_count = 0
        self.last_breach = None

    def check(self, **kwargs) -> Optional[BreacherEvent]:
        """Check if circuit breaker should trigger."""
        if not self.enabled:
            return None
        return self._evaluate(**kwargs)

    def _evaluate(self, **kwargs) -> Optional[BreacherEvent]:
        """Override in subclasses."""
        raise NotImplementedError

class DrawdownCircuitBreaker(BaseCircuitBreaker):
    """Maximum drawdown circuit breaker.

    Triggers when portfolio drawdown exceeds threshold.
    """

    def __init__(self,
                 max_drawdown: float = 0.15,  # 15% max drawdown
                 warning_drawdown: float = 0.10,  # 10% warning
                 enabled: bool = True) -> None:
        super().__init__("DrawdownBreaker", enabled)
        self.max_drawdown = max_drawdown
        self.warning_drawdown = warning_drawdown

    def _evaluate(self, portfolio_value=None, value_series=None, **kwargs) -> Optional[BreacherEvent]:
        """Evaluate drawdown breach."""
        # Use value_series if provided, otherwise try to use portfolio_value as series
        if value_series is not None:
            portfolio_value = value_series
        elif isinstance(portfolio_value, (int, float)):
            return None  # Need time series for drawdown calculation

        if portfolio_value is None or len(portfolio_value) < 2:
            return None

        peak = portfolio_value.expanding().max()
        current_dd = (portfolio_value.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]

        if current_dd <= -self.max_drawdown:
            self.breach_count += 1
            self.last_breach = datetime.now()
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.EMERGENCY,
                value=current_dd,
                threshold=-self.max_drawdown,
                message=f"Portfolio drawdown {current_dd:.1%} exceeds max {self.max_drawdown:.1%}",
                action_taken="HALT_ALL_TRADING",
            )
        elif current_dd <= -self.warning_drawdown:
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.WARNING,
                value=current_dd,
                threshold=-self.warning_drawdown,
                message=f"Portfolio drawdown {current_dd:.1%} exceeds warning {self.warning_drawdown:.1%}",
                action_taken="REDUCE_POSITION_SIZE",
            )

        return None

class VolatilityCircuitBreaker(BaseCircuitBreaker):
    """Volatility spike circuit breaker.

    Triggers when volatility exceeds normal ranges.
    """

    def __init__(self,
                 volatility_threshold: float = 3.0,  # 3x normal volatility
                 lookback_window: int = 20,
                 enabled: bool = True) -> None:
        super().__init__("VolatilityBreaker", enabled)
        self.volatility_threshold = volatility_threshold
        self.lookback_window = lookback_window

    def _evaluate(self, returns: pd.Series, **kwargs) -> Optional[BreacherEvent]:
        """Evaluate volatility breach."""
        if len(returns) < self.lookback_window + 1:
            return None

        # Calculate rolling volatility
        recent_vol = returns.tail(5).std() * np.sqrt(252)  # Annualized
        normal_vol = returns.tail(self.lookback_window).std() * np.sqrt(252)

        vol_ratio = recent_vol / normal_vol if normal_vol > 0 else 0

        if vol_ratio > self.volatility_threshold:
            self.breach_count += 1
            self.last_breach = datetime.now()
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.CRITICAL,
                value=vol_ratio,
                threshold=self.volatility_threshold,
                message=f"Volatility spike {vol_ratio:.1f}x normal levels",
                action_taken="PAUSE_NEW_POSITIONS",
            )

        return None

class DailyLossCircuitBreaker(BaseCircuitBreaker):
    """Daily loss limit circuit breaker.

    Triggers when daily losses exceed threshold.
    """

    def __init__(self,
                 max_daily_loss: float = 0.05,  # 5% max daily loss
                 enabled: bool = True) -> None:
        super().__init__("DailyLossBreaker", enabled)
        self.max_daily_loss = max_daily_loss

    def _evaluate(self, portfolio_value=None, value_series=None, returns=None, **kwargs) -> Optional[BreacherEvent]:
        """Evaluate daily loss breach."""
        # Use returns directly if provided
        if returns is not None and len(returns) > 0:
            daily_return = returns.iloc[-1]
        else:
            # Try to calculate from portfolio value series
            if value_series is not None:
                portfolio_value = value_series
            elif isinstance(portfolio_value, (int, float)):
                return None

            if portfolio_value is None or len(portfolio_value) < 2:
                return None

            daily_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[-2]) - 1

        if daily_return <= -self.max_daily_loss:
            self.breach_count += 1
            self.last_breach = datetime.now()
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.EMERGENCY,
                value=daily_return,
                threshold=-self.max_daily_loss,
                message=f"Daily loss {daily_return:.1%} exceeds limit {self.max_daily_loss:.1%}",
                action_taken="HALT_ALL_TRADING",
            )

        return None

class PositionConcentrationBreaker(BaseCircuitBreaker):
    """Position concentration circuit breaker.

    Triggers when single position becomes too large.
    """

    def __init__(self,
                 max_position_weight: float = 0.30,  # 30% max per position
                 enabled: bool = True) -> None:
        super().__init__("ConcentrationBreaker", enabled)
        self.max_position_weight = max_position_weight

    def _evaluate(self, position_weights: Optional[Dict[str, float]] = None, positions: Optional[Dict[str, float]] = None, **kwargs) -> Optional[BreacherEvent]:
        """Evaluate concentration breach."""
        # Use positions if position_weights not provided
        if position_weights is None:
            position_weights = positions

        if not position_weights:
            return None

        max_weight = max(abs(weight) for weight in position_weights.values())
        max_symbol = max(position_weights.keys(),
                        key=lambda k: abs(position_weights[k]))

        if max_weight > self.max_position_weight:
            self.breach_count += 1
            self.last_breach = datetime.now()
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.WARNING,
                value=max_weight,
                threshold=self.max_position_weight,
                message=f"Position {max_symbol} weight {max_weight:.1%} exceeds limit {self.max_position_weight:.1%}",
                action_taken="REBALANCE_PORTFOLIO",
            )

        return None

class CorrelationBreaker(BaseCircuitBreaker):
    """Correlation breakdown circuit breaker.

    Triggers when asset correlations spike unexpectedly.
    """

    def __init__(self,
                 correlation_threshold: float = 0.8,  # 80% correlation threshold
                 lookback_window: int = 20,
                 enabled: bool = True) -> None:
        super().__init__("CorrelationBreaker", enabled)
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window

    def _evaluate(self, returns_matrix: Optional[pd.DataFrame] = None, **kwargs) -> Optional[BreacherEvent]:
        """Evaluate correlation breach."""
        if returns_matrix is None or len(returns_matrix) < self.lookback_window:
            return None

        recent_corr = returns_matrix.tail(self.lookback_window).corr()

        # Find maximum off-diagonal correlation
        corr_matrix = recent_corr.values
        np.fill_diagonal(corr_matrix, 0)  # Remove diagonal
        max_corr = np.max(np.abs(corr_matrix))

        if max_corr > self.correlation_threshold:
            self.breach_count += 1
            self.last_breach = datetime.now()
            return BreacherEvent(
                timestamp=datetime.now(),
                breaker_name=self.name,
                severity=BreacherSeverity.WARNING,
                value=max_corr,
                threshold=self.correlation_threshold,
                message=f"Asset correlation {max_corr:.1%} exceeds threshold {self.correlation_threshold:.1%}",
                action_taken="DIVERSIFY_POSITIONS",
            )

        return None

class CircuitBreakerManager:
    """Manages multiple circuit breakers and coordinates responses.

    Provides centralized monitoring and action coordination.
    """

    def __init__(self) -> None:
        """Initialize circuit breaker manager."""
        self.breakers = {}
        self.breach_history = []
        self.active_halts = set()

        # Initialize default breakers
        self._setup_default_breakers()

    def _setup_default_breakers(self) -> None:
        """Setup default institutional circuit breakers."""
        self.register_breaker(DrawdownCircuitBreaker())
        self.register_breaker(VolatilityCircuitBreaker())
        self.register_breaker(DailyLossCircuitBreaker())
        self.register_breaker(PositionConcentrationBreaker())
        self.register_breaker(CorrelationBreaker())

    def register_breaker(self, breaker: BaseCircuitBreaker) -> None:
        """Register a new circuit breaker."""
        self.breakers[breaker.name] = breaker

    def check_all_breakers(self, **market_data) -> List[BreacherEvent]:
        """Check all circuit breakers with current market data.

        Args:
        ----
            **market_data: Market data for evaluation

        Returns:
        -------
            List of breacher events

        """
        events = []

        for breaker in self.breakers.values():
            event = breaker.check(**market_data)
            if event:
                events.append(event)
                self.breach_history.append(event)

                # Handle emergency actions
                if event.severity == BreacherSeverity.EMERGENCY:
                    self.active_halts.add(event.breaker_name)

        return events

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status based on circuit breaker states.

        Returns
        -------
            Trading status information

        """
        emergency_halts = list(self.active_halts)

        return {
            "trading_allowed": len(emergency_halts) == 0,
            "active_halts": emergency_halts,
            "total_breaches_today": len([e for e in self.breach_history
                                        if e.timestamp.date() == datetime.now().date()]),
            "risk_level": self._calculate_risk_level(),
        }


    def clear_halt(self, breaker_name: str) -> None:
        """Clear a specific trading halt."""
        self.active_halts.discard(breaker_name)

    def clear_all_halts(self) -> None:
        """Clear all trading halts (use with caution)."""
        self.active_halts.clear()

    def get_breach_summary(self, days: int = 7) -> pd.DataFrame:
        """Get breach summary for the last N days.

        Args:
        ----
            days: Number of days to look back

        Returns:
        -------
            DataFrame with breach summary

        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_breaches = [e for e in self.breach_history if e.timestamp >= cutoff_date]

        if not recent_breaches:
            return pd.DataFrame()

        breach_data = []
        for event in recent_breaches:
            breach_data.append({
                "timestamp": event.timestamp,
                "breaker": event.breaker_name,
                "severity": event.severity.value,
                "value": event.value,
                "threshold": event.threshold,
                "message": event.message,
            })

        return pd.DataFrame(breach_data)

    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level."""
        recent_breaches = len([e for e in self.breach_history
                             if e.timestamp >= datetime.now() - timedelta(hours=24)])

        if len(self.active_halts) > 0:
            return "CRITICAL"
        elif recent_breaches >= 5:
            return "HIGH"
        elif recent_breaches >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def get_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}

        for name, breaker in self.breakers.items():
            status[name] = {
                "enabled": breaker.enabled,
                "breach_count": breaker.breach_count,
                "last_breach": breaker.last_breach,
                "is_halted": name in self.active_halts,
            }

        return status
