"""
Risk Monitor - Astra Trading Platform
=====================================

Real-time portfolio risk monitoring and alerting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from .circuit_breakers import CircuitBreakerManager, BreacherEvent
from .position_sizing import PositionSizer

@dataclass
class RiskMetrics:
    """Portfolio risk metrics snapshot."""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    volatility: float
    max_drawdown: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    position_count: int
    largest_position: float
    total_exposure: float
    beta: float
    correlation_risk: float

@dataclass
class RiskAlert:
    """Risk monitoring alert."""
    timestamp: datetime
    alert_type: str
    severity: str
    metric: str
    current_value: float
    threshold: float
    message: str
    recommended_action: str

class RiskMonitor:
    """
    Real-time portfolio risk monitoring system.
    
    Integrates circuit breakers, position sizing, and risk metrics
    for comprehensive portfolio oversight.
    """
    
    def __init__(self):
        """Initialize risk monitor."""
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.position_sizer = PositionSizer()
        self.risk_history = []
        self.alert_history = []
        
        # Risk thresholds
        self.thresholds = {
            'max_drawdown': 0.15,
            'daily_var_95': 0.05,
            'volatility': 0.25,
            'concentration': 0.30,
            'total_exposure': 1.0,
            'correlation': 0.80
        }
        
    def update_risk_metrics(self, 
                           portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """
        Update and calculate current risk metrics.
        
        Args:
            portfolio_data: Dictionary containing portfolio information
            
        Returns:
            Current risk metrics
        """
        portfolio_value = portfolio_data.get('portfolio_value', 0)
        returns = portfolio_data.get('returns', pd.Series())
        positions = portfolio_data.get('positions', {})
        prices = portfolio_data.get('prices', pd.Series())
        
        # Calculate metrics
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_return=self._calculate_total_return(portfolio_value, portfolio_data.get('initial_value', 100000)),
            daily_return=returns.iloc[-1] if len(returns) > 0 else 0.0,
            volatility=returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0,
            max_drawdown=self._calculate_max_drawdown(portfolio_data.get('value_series', pd.Series())),
            var_95=np.percentile(returns, 5) if len(returns) > 0 else 0.0,
            var_99=np.percentile(returns, 1) if len(returns) > 0 else 0.0,
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            position_count=len(positions),
            largest_position=max(abs(w) for w in positions.values()) if positions else 0.0,
            total_exposure=sum(abs(w) for w in positions.values()) if positions else 0.0,
            beta=self._calculate_beta(returns, portfolio_data.get('benchmark_returns', pd.Series())),
            correlation_risk=self._calculate_correlation_risk(portfolio_data.get('returns_matrix', pd.DataFrame()))
        )
        
        # Store metrics
        self.risk_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
            
        return metrics
    
    def check_risk_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """
        Check for risk threshold breaches and generate alerts.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            List of risk alerts
        """
        alerts = []
        
        # Check drawdown
        if abs(metrics.max_drawdown) > self.thresholds['max_drawdown']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="DRAWDOWN_BREACH",
                severity="HIGH",
                metric="max_drawdown",
                current_value=metrics.max_drawdown,
                threshold=self.thresholds['max_drawdown'],
                message=f"Maximum drawdown {metrics.max_drawdown:.1%} exceeds threshold {self.thresholds['max_drawdown']:.1%}",
                recommended_action="REDUCE_POSITION_SIZE"
            ))
        
        # Check volatility
        if metrics.volatility > self.thresholds['volatility']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="VOLATILITY_SPIKE",
                severity="MEDIUM",
                metric="volatility",
                current_value=metrics.volatility,
                threshold=self.thresholds['volatility'],
                message=f"Portfolio volatility {metrics.volatility:.1%} exceeds threshold {self.thresholds['volatility']:.1%}",
                recommended_action="REVIEW_POSITIONS"
            ))
        
        # Check concentration
        if metrics.largest_position > self.thresholds['concentration']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="CONCENTRATION_RISK",
                severity="MEDIUM",
                metric="largest_position",
                current_value=metrics.largest_position,
                threshold=self.thresholds['concentration'],
                message=f"Largest position {metrics.largest_position:.1%} exceeds concentration limit {self.thresholds['concentration']:.1%}",
                recommended_action="REBALANCE_PORTFOLIO"
            ))
        
        # Check total exposure
        if metrics.total_exposure > self.thresholds['total_exposure']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="EXCESSIVE_EXPOSURE",
                severity="HIGH",
                metric="total_exposure",
                current_value=metrics.total_exposure,
                threshold=self.thresholds['total_exposure'],
                message=f"Total exposure {metrics.total_exposure:.1%} exceeds limit {self.thresholds['total_exposure']:.1%}",
                recommended_action="REDUCE_EXPOSURE"
            ))
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def run_comprehensive_risk_check(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive risk check including circuit breakers and metrics.
        
        Args:
            portfolio_data: Portfolio data for analysis
            
        Returns:
            Comprehensive risk assessment
        """
        # Update risk metrics
        metrics = self.update_risk_metrics(portfolio_data)
        
        # Check risk alerts
        alerts = self.check_risk_alerts(metrics)
        
        # Check circuit breakers
        breaker_events = self.circuit_breaker_manager.check_all_breakers(**portfolio_data)
        
        # Get trading status
        trading_status = self.circuit_breaker_manager.get_trading_status()
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(metrics, alerts, breaker_events)
        
        return {
            'risk_metrics': metrics,
            'alerts': alerts,
            'circuit_breaker_events': breaker_events,
            'trading_status': trading_status,
            'risk_score': risk_score,
            'recommendations': self._generate_recommendations(metrics, alerts, breaker_events)
        }
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """
        Get risk dashboard data for visualization.
        
        Returns:
            Risk dashboard data
        """
        if not self.risk_history:
            return {}
        
        latest_metrics = self.risk_history[-1]
        recent_alerts = [a for a in self.alert_history 
                        if a.timestamp >= datetime.now() - timedelta(days=1)]
        
        # Calculate trends
        if len(self.risk_history) >= 2:
            prev_metrics = self.risk_history[-2]
            trends = {
                'portfolio_value': latest_metrics.portfolio_value - prev_metrics.portfolio_value,
                'volatility': latest_metrics.volatility - prev_metrics.volatility,
                'max_drawdown': latest_metrics.max_drawdown - prev_metrics.max_drawdown
            }
        else:
            trends = {'portfolio_value': 0, 'volatility': 0, 'max_drawdown': 0}
        
        return {
            'current_metrics': latest_metrics,
            'recent_alerts': recent_alerts,
            'trends': trends,
            'circuit_breaker_status': self.circuit_breaker_manager.get_breaker_status(),
            'trading_status': self.circuit_breaker_manager.get_trading_status(),
            'alert_count_24h': len(recent_alerts)
        }
    
    def _calculate_total_return(self, current_value: float, initial_value: float) -> float:
        """Calculate total return."""
        return (current_value / initial_value) - 1 if initial_value > 0 else 0.0
    
    def _calculate_max_drawdown(self, value_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(value_series) < 2:
            return 0.0
        
        peak = value_series.expanding().max()
        drawdown = (value_series - peak) / peak
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta."""
        if len(portfolio_returns) < 10 or len(benchmark_returns) < 10:
            return 1.0  # Default beta
        
        # Align series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 10:
            return 1.0
        
        portfolio_ret = aligned_data.iloc[:, 0]
        benchmark_ret = aligned_data.iloc[:, 1]
        
        covariance = np.cov(portfolio_ret, benchmark_ret)[0, 1]
        benchmark_variance = np.var(benchmark_ret)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_correlation_risk(self, returns_matrix: pd.DataFrame) -> float:
        """Calculate correlation risk metric."""
        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            return 0.0
        
        correlation_matrix = returns_matrix.corr()
        
        # Get off-diagonal correlations
        corr_values = correlation_matrix.values
        np.fill_diagonal(corr_values, 0)
        
        # Return maximum absolute correlation
        return np.max(np.abs(corr_values))
    
    def _calculate_risk_score(self, 
                             metrics: RiskMetrics, 
                             alerts: List[RiskAlert], 
                             breaker_events: List[BreacherEvent]) -> int:
        """
        Calculate overall risk score (0-100).
        
        Higher scores indicate higher risk.
        """
        score = 0
        
        # Base score from metrics
        if abs(metrics.max_drawdown) > 0.10:
            score += 20
        if metrics.volatility > 0.20:
            score += 15
        if metrics.largest_position > 0.25:
            score += 10
        if metrics.var_95 < -0.05:
            score += 15
        
        # Alert contributions
        for alert in alerts:
            if alert.severity == "HIGH":
                score += 15
            elif alert.severity == "MEDIUM":
                score += 10
            else:
                score += 5
        
        # Circuit breaker contributions
        for event in breaker_events:
            if event.severity.value == "EMERGENCY":
                score += 25
            elif event.severity.value == "CRITICAL":
                score += 15
            else:
                score += 5
        
        return min(score, 100)  # Cap at 100
    
    def _generate_recommendations(self, 
                                 metrics: RiskMetrics,
                                 alerts: List[RiskAlert],
                                 breaker_events: List[BreacherEvent]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if abs(metrics.max_drawdown) > 0.10:
            recommendations.append("Consider reducing position sizes to limit further drawdown")
        
        if metrics.volatility > 0.20:
            recommendations.append("Portfolio volatility is elevated - review position diversification")
        
        if metrics.largest_position > 0.25:
            recommendations.append("High position concentration detected - consider rebalancing")
        
        if len(alerts) > 3:
            recommendations.append("Multiple risk alerts active - conduct comprehensive portfolio review")
        
        if len(breaker_events) > 0:
            recommendations.append("Circuit breaker events detected - review risk management settings")
        
        if not recommendations:
            recommendations.append("Risk levels within acceptable ranges - maintain current monitoring")
        
        return recommendations