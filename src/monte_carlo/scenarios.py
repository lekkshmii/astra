"""
Monte Carlo Scenarios - Astra Trading Platform
==============================================

Python wrappers for Rust Monte Carlo scenarios with fallback implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ScenarioResult:
    """Results from a Monte Carlo scenario."""
    paths: np.ndarray
    statistics: Dict[str, float]
    percentiles: Dict[str, float]
    scenario_params: Dict[str, Any]

class BaseScenario(ABC):
    """Base class for Monte Carlo scenarios."""
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
    
    @abstractmethod
    def run(self, 
            initial_prices: List[float], 
            num_simulations: int, 
            time_steps: int) -> ScenarioResult:
        """Run the Monte Carlo scenario."""
        pass

class FlashCrashScenario(BaseScenario):
    """
    Flash crash scenario simulation.
    
    Models sudden market liquidity evaporation events.
    """
    
    def __init__(self, 
                 crash_probability: float = 0.001,
                 crash_magnitude: float = 0.15,
                 recovery_rate: float = 0.1,
                 **kwargs):
        super().__init__("FlashCrash", 
                        crash_probability=crash_probability,
                        crash_magnitude=crash_magnitude,
                        recovery_rate=recovery_rate,
                        **kwargs)
        self.crash_probability = crash_probability
        self.crash_magnitude = crash_magnitude
        self.recovery_rate = recovery_rate
    
    def run(self, 
            initial_prices: List[float], 
            num_simulations: int = 10000, 
            time_steps: int = 252) -> ScenarioResult:
        """Run flash crash simulation."""
        
        try:
            # Try Rust implementation first
            import astra_monte_carlo as mc
            paths = mc.monte_carlo_flash_crash(
                initial_prices, num_simulations, time_steps,
                self.crash_probability, self.crash_magnitude, self.recovery_rate
            )
            paths = np.array(paths)
            
        except ImportError:
            # Fallback to Python implementation
            paths = self._python_implementation(
                initial_prices, num_simulations, time_steps
            )
        
        # Calculate statistics
        final_prices = paths[:, -1]
        statistics = {
            'mean_final_price': np.mean(final_prices),
            'std_final_price': np.std(final_prices),
            'crash_frequency': self._estimate_crash_frequency(paths),
            'max_drawdown_mean': np.mean([self._max_drawdown(path) for path in paths])
        }
        
        percentiles = {
            'p5': np.percentile(final_prices, 5),
            'p25': np.percentile(final_prices, 25),
            'p50': np.percentile(final_prices, 50),
            'p75': np.percentile(final_prices, 75),
            'p95': np.percentile(final_prices, 95)
        }
        
        return ScenarioResult(
            paths=paths,
            statistics=statistics,
            percentiles=percentiles,
            scenario_params=self.params
        )
    
    def _python_implementation(self, 
                              initial_prices: List[float], 
                              num_simulations: int, 
                              time_steps: int) -> np.ndarray:
        """Python fallback implementation."""
        np.random.seed(42)
        paths = []
        
        for _ in range(num_simulations):
            path = [initial_prices[0]]
            in_crash = False
            crash_remaining = 0
            
            for t in range(1, time_steps):
                if not in_crash and np.random.random() < self.crash_probability:
                    in_crash = True
                    crash_remaining = 10
                
                if in_crash:
                    crash_remaining -= 1
                    if crash_remaining <= 0:
                        in_crash = False
                    return_rate = -self.crash_magnitude + np.random.normal(0, 0.1)
                else:
                    return_rate = np.random.normal(0, 0.02)
                
                path.append(path[-1] * (1 + return_rate))
            
            paths.append(path)
        
        return np.array(paths)
    
    def _estimate_crash_frequency(self, paths: np.ndarray) -> float:
        """Estimate crash frequency from simulation paths."""
        crash_count = 0
        for path in paths:
            returns = np.diff(path) / path[:-1]
            if np.any(returns < -self.crash_magnitude * 0.5):
                crash_count += 1
        return crash_count / len(paths)
    
    def _max_drawdown(self, path: np.ndarray) -> float:
        """Calculate maximum drawdown for a price path."""
        peak = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        return np.min(drawdown)

class RegimeSwitchingScenario(BaseScenario):
    """
    Regime switching scenario simulation.
    
    Models transitions between bull and bear market regimes.
    """
    
    def __init__(self,
                 bull_volatility: float = 0.15,
                 bear_volatility: float = 0.35,
                 transition_probability: float = 0.02,
                 **kwargs):
        super().__init__("RegimeSwitching",
                        bull_volatility=bull_volatility,
                        bear_volatility=bear_volatility,
                        transition_probability=transition_probability,
                        **kwargs)
        self.bull_volatility = bull_volatility
        self.bear_volatility = bear_volatility
        self.transition_probability = transition_probability
    
    def run(self,
            initial_prices: List[float],
            num_simulations: int = 10000,
            time_steps: int = 252) -> ScenarioResult:
        """Run regime switching simulation."""
        
        try:
            # Try Rust implementation
            import astra_monte_carlo as mc
            paths = mc.monte_carlo_regime_switching(
                initial_prices, num_simulations, time_steps,
                self.bull_volatility, self.bear_volatility, self.transition_probability
            )
            paths = np.array(paths)
            
        except ImportError:
            # Python fallback
            paths = self._python_implementation(
                initial_prices, num_simulations, time_steps
            )
        
        final_prices = paths[:, -1]
        statistics = {
            'mean_final_price': np.mean(final_prices),
            'std_final_price': np.std(final_prices),
            'regime_transitions_mean': self._estimate_transitions(paths)
        }
        
        percentiles = {
            'p5': np.percentile(final_prices, 5),
            'p25': np.percentile(final_prices, 25),
            'p50': np.percentile(final_prices, 50),
            'p75': np.percentile(final_prices, 75),
            'p95': np.percentile(final_prices, 95)
        }
        
        return ScenarioResult(
            paths=paths,
            statistics=statistics,
            percentiles=percentiles,
            scenario_params=self.params
        )
    
    def _python_implementation(self,
                              initial_prices: List[float],
                              num_simulations: int,
                              time_steps: int) -> np.ndarray:
        """Python fallback implementation."""
        np.random.seed(42)
        paths = []
        
        for _ in range(num_simulations):
            path = [initial_prices[0]]
            is_bull_market = True
            
            for t in range(1, time_steps):
                if np.random.random() < self.transition_probability:
                    is_bull_market = not is_bull_market
                
                volatility = self.bull_volatility if is_bull_market else self.bear_volatility
                drift = 0.0008 if is_bull_market else -0.0002
                
                return_rate = drift + np.random.normal(0, volatility)
                path.append(path[-1] * (1 + return_rate))
            
            paths.append(path)
        
        return np.array(paths)
    
    def _estimate_transitions(self, paths: np.ndarray) -> float:
        """Estimate average number of regime transitions."""
        # Simplified estimate based on volatility changes
        return self.transition_probability * len(paths[0])

class CorrelationBreakdownScenario(BaseScenario):
    """
    Correlation breakdown scenario.
    
    Models how asset correlations change during stress periods.
    """
    
    def __init__(self,
                 normal_correlation: float = 0.3,
                 stress_correlation: float = 0.8,
                 stress_threshold: float = 2.0,
                 **kwargs):
        super().__init__("CorrelationBreakdown",
                        normal_correlation=normal_correlation,
                        stress_correlation=stress_correlation,
                        stress_threshold=stress_threshold,
                        **kwargs)
        self.normal_correlation = normal_correlation
        self.stress_correlation = stress_correlation
        self.stress_threshold = stress_threshold
    
    def run(self,
            initial_prices: List[float],
            num_simulations: int = 10000,
            time_steps: int = 252) -> ScenarioResult:
        """Run correlation breakdown simulation."""
        
        try:
            # Try Rust implementation
            import astra_monte_carlo as mc
            paths = mc.monte_carlo_correlation_breakdown(
                initial_prices, num_simulations, time_steps,
                self.normal_correlation, self.stress_correlation, self.stress_threshold
            )
            paths = np.array(paths)
            
        except ImportError:
            # Python fallback - multi-asset simulation
            paths = self._python_implementation(
                initial_prices, num_simulations, time_steps
            )
        
        # Calculate multi-asset statistics
        final_prices = paths[:, :, -1]  # shape: (simulations, assets, time)
        statistics = {
            'mean_final_prices': np.mean(final_prices, axis=0).tolist(),
            'correlation_stress_frequency': self._estimate_stress_frequency(paths)
        }
        
        percentiles = {}
        for i in range(len(initial_prices)):
            asset_finals = final_prices[:, i]
            percentiles[f'asset_{i}_p5'] = np.percentile(asset_finals, 5)
            percentiles[f'asset_{i}_p95'] = np.percentile(asset_finals, 95)
        
        return ScenarioResult(
            paths=paths,
            statistics=statistics,
            percentiles=percentiles,
            scenario_params=self.params
        )
    
    def _python_implementation(self,
                              initial_prices: List[float],
                              num_simulations: int,
                              time_steps: int) -> np.ndarray:
        """Python fallback implementation."""
        np.random.seed(42)
        num_assets = len(initial_prices)
        paths = []
        
        for _ in range(num_simulations):
            asset_paths = [[price] for price in initial_prices]
            
            for t in range(1, time_steps):
                market_shock = abs(np.random.normal(0, 1))
                correlation = (self.stress_correlation if market_shock > self.stress_threshold 
                             else self.normal_correlation)
                
                common_factor = np.random.normal(0, 1) * np.sqrt(correlation)
                
                for i in range(num_assets):
                    idiosyncratic = np.random.normal(0, 1) * np.sqrt(1 - correlation)
                    return_rate = (common_factor + idiosyncratic) * 0.02
                    asset_paths[i].append(asset_paths[i][-1] * (1 + return_rate))
            
            paths.append(asset_paths)
        
        return np.array(paths)
    
    def _estimate_stress_frequency(self, paths: np.ndarray) -> float:
        """Estimate frequency of stress correlation periods."""
        return 0.1  # Placeholder

class ScenarioRunner:
    """
    Monte Carlo scenario runner.
    
    Orchestrates multiple scenarios and portfolio stress testing.
    """
    
    def __init__(self):
        self.scenarios = {}
        
        # Register default scenarios
        self.register_scenario('flash_crash', FlashCrashScenario())
        self.register_scenario('regime_switching', RegimeSwitchingScenario())
        self.register_scenario('correlation_breakdown', CorrelationBreakdownScenario())
    
    def register_scenario(self, name: str, scenario: BaseScenario):
        """Register a scenario for running."""
        self.scenarios[name] = scenario
    
    def run_scenario(self, 
                    scenario_name: str, 
                    initial_prices: List[float],
                    num_simulations: int = 10000,
                    time_steps: int = 252) -> ScenarioResult:
        """Run a specific scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        return self.scenarios[scenario_name].run(
            initial_prices, num_simulations, time_steps
        )
    
    def run_all_scenarios(self,
                         initial_prices: List[float],
                         num_simulations: int = 10000,
                         time_steps: int = 252) -> Dict[str, ScenarioResult]:
        """Run all registered scenarios."""
        results = {}
        
        for name, scenario in self.scenarios.items():
            print(f"Running {name} scenario...")
            results[name] = scenario.run(initial_prices, num_simulations, time_steps)
            
        return results
    
    def portfolio_stress_test(self,
                             portfolio_weights: List[float],
                             asset_prices: List[float],
                             scenarios: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Run portfolio stress test across scenarios.
        
        Args:
            portfolio_weights: Portfolio allocation weights
            asset_prices: Current asset prices
            scenarios: List of scenarios to run (None = all)
            
        Returns:
            Stress test results by scenario
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        stress_results = {}
        
        for scenario_name in scenarios:
            result = self.run_scenario(scenario_name, asset_prices)
            
            # Calculate portfolio-level metrics
            if len(portfolio_weights) == len(asset_prices):
                portfolio_paths = self._calculate_portfolio_paths(
                    result.paths, portfolio_weights
                )
                
                portfolio_final = portfolio_paths[:, -1]
                portfolio_returns = (portfolio_final / portfolio_paths[:, 0]) - 1
                
                stress_results[scenario_name] = {
                    'var_95': np.percentile(portfolio_returns, 5),
                    'var_99': np.percentile(portfolio_returns, 1),
                    'expected_return': np.mean(portfolio_returns),
                    'volatility': np.std(portfolio_returns),
                    'max_loss': np.min(portfolio_returns)
                }
        
        return stress_results
    
    def _calculate_portfolio_paths(self, 
                                  asset_paths: np.ndarray, 
                                  weights: List[float]) -> np.ndarray:
        """Calculate portfolio value paths from asset paths."""
        if len(asset_paths.shape) == 2:
            # Single asset case
            return asset_paths * weights[0]
        else:
            # Multi-asset case
            portfolio_paths = np.zeros((asset_paths.shape[0], asset_paths.shape[2]))
            
            for i, weight in enumerate(weights):
                if i < asset_paths.shape[1]:
                    portfolio_paths += asset_paths[:, i, :] * weight
                    
            return portfolio_paths