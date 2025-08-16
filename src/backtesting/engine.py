"""Backtesting Engine - Astra Trading Platform.
===========================================

High-performance backtesting engine built on pandas/numpy.
Will integrate with VectorBT once Python environment is compatible.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    portfolio_value: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

    def total_return(self) -> float:
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def max_drawdown(self) -> float:
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def calmar_ratio(self) -> float:
        annual_return = self.total_return()
        max_dd = abs(self.max_drawdown())
        return annual_return / max_dd if max_dd > 0 else 0

class BacktestEngine:
    """Professional backtesting engine for strategy evaluation.

    Features:
    - Signal-based backtesting
    - Transaction cost modeling
    - Portfolio rebalancing
    - Performance metrics calculation
    """

    def __init__(self, initial_capital: float = 100000, commission: float = 0.001) -> None:
        """Initialize backtest engine.

        Args:
        ----
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 10 basis points)

        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 1.0,
        rebalance_freq: str = "D",
    ) -> BacktestResult:
        """Run backtest with given data and signals.

        Args:
        ----
            data: Price data (adjusted close)
            signals: Trading signals (-1, 0, 1 for short, neutral, long)
            position_size: Position sizing (1.0 = full capital)
            rebalance_freq: Rebalancing frequency

        Returns:
        -------
            BacktestResult with portfolio performance

        """
        if data.empty or signals.empty:
            msg = "Data and signals cannot be empty"
            raise ValueError(msg)

        # Align data and signals
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]

        # Calculate returns
        data.pct_change().fillna(0)

        # Initialize portfolio tracking
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        positions = {}
        trades = []

        # Track portfolio returns
        portfolio_returns = []

        for i, (date, signal_row) in enumerate(signals.iterrows()):
            if i == 0:
                portfolio_returns.append(0.0)
                continue

            # Calculate current portfolio value
            current_prices = data.loc[date]
            portfolio_val = cash

            for symbol, shares in positions.items():
                if symbol in current_prices:
                    portfolio_val += shares * current_prices[symbol]

            # Rebalance based on signals
            if rebalance_freq == "D" or i % 5 == 0:  # Daily or weekly
                new_positions = self._rebalance_portfolio(
                    signal_row, current_prices, portfolio_val, position_size,
                )

                # Execute trades
                trades_made = self._execute_trades(
                    date, positions, new_positions, current_prices,
                )
                trades.extend(trades_made)

                # Update cash and positions
                cash, positions = self._update_portfolio(
                    cash, positions, new_positions, current_prices,
                )

            # Calculate portfolio return
            prev_val = portfolio_value[-1]
            current_val = cash
            for symbol, shares in positions.items():
                if symbol in current_prices:
                    current_val += shares * current_prices[symbol]

            portfolio_return = (current_val - prev_val) / prev_val
            portfolio_returns.append(portfolio_return)
            portfolio_value.append(current_val)

        # Create results - align lengths properly
        portfolio_series = pd.Series(portfolio_value[1:], index=common_index[1:])
        returns_series = pd.Series(portfolio_returns[1:], index=common_index[1:])
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        metrics = self._calculate_metrics(portfolio_series, returns_series)

        return BacktestResult(
            portfolio_value=portfolio_series,
            returns=returns_series,
            trades=trades_df,
            metrics=metrics,
        )

    def _rebalance_portfolio(
        self,
        signals: pd.Series,
        prices: pd.Series,
        portfolio_value: float,
        position_size: float,
    ) -> Dict[str, float]:
        """Calculate target positions based on signals."""
        target_positions = {}

        # Calculate target allocation per symbol
        active_signals = signals[signals != 0]
        if len(active_signals) == 0:
            return {}

        allocation_per_symbol = portfolio_value * position_size / len(active_signals)

        for symbol, signal in active_signals.items():
            if symbol in prices and prices[symbol] > 0:
                target_value = allocation_per_symbol * signal
                target_shares = target_value / prices[symbol]
                target_positions[symbol] = target_shares

        return target_positions

    def _execute_trades(
        self,
        date: datetime,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        prices: pd.Series,
    ) -> List[Dict]:
        """Execute trades to reach target positions."""
        trades = []

        all_symbols = set(current_positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_shares = current_positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            shares_delta = target_shares - current_shares

            if abs(shares_delta) > 1e-6 and symbol in prices:  # Minimum trade size
                trade_value = shares_delta * prices[symbol]
                commission_cost = abs(trade_value) * self.commission

                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "shares": shares_delta,
                    "price": prices[symbol],
                    "value": trade_value,
                    "commission": commission_cost,
                })

        return trades

    def _update_portfolio(
        self,
        cash: float,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        prices: pd.Series,
    ) -> Tuple[float, Dict[str, float]]:
        """Update cash and positions after trades."""
        new_cash = cash
        new_positions = current_positions.copy()

        all_symbols = set(current_positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_shares = current_positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            shares_delta = target_shares - current_shares

            if abs(shares_delta) > 1e-6 and symbol in prices:
                trade_value = shares_delta * prices[symbol]
                commission_cost = abs(trade_value) * self.commission

                new_cash -= trade_value + commission_cost
                new_positions[symbol] = target_shares

        # Remove zero positions
        new_positions = {k: v for k, v in new_positions.items() if abs(v) > 1e-6}

        return new_cash, new_positions

    def _calculate_metrics(
        self,
        portfolio_value: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1

        # Annualized metrics
        trading_days = len(returns)
        annual_factor = 252 / trading_days if trading_days > 0 else 1

        annual_return = (1 + total_return) ** annual_factor - 1
        volatility = returns.std() * np.sqrt(252)

        # Risk metrics
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "final_value": portfolio_value.iloc[-1],
        }
