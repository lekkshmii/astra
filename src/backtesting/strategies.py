"""Trading Strategies - Astra Trading Platform.
===========================================

Collection of trading strategies for backtesting.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, **params) -> None:
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data."""

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"

class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy using moving average crossover.

    Signals:
    - Buy (1) when short MA > long MA
    - Sell (-1) when short MA < long MA
    - Hold (0) otherwise
    """

    def __init__(self, short_window: int = 10, long_window: int = 30, **kwargs) -> None:
        super().__init__("MomentumStrategy", short_window=short_window, long_window=long_window, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            prices = data[symbol].dropna()
            if len(prices) < self.long_window:
                continue

            # Calculate moving averages
            short_ma = prices.rolling(self.short_window).mean()
            long_ma = prices.rolling(self.long_window).mean()

            # Generate signals
            signals.loc[short_ma.index, symbol] = np.where(
                short_ma > long_ma, 1.0,
                np.where(short_ma < long_ma, -1.0, 0.0),
            )

        return signals.fillna(0.0)

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using z-score.

    Signals:
    - Buy (1) when z-score < -threshold (oversold)
    - Sell (-1) when z-score > threshold (overbought)
    - Hold (0) otherwise
    """

    def __init__(self, window: int = 20, threshold: float = 2.0, **kwargs) -> None:
        super().__init__("MeanReversionStrategy", window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            prices = data[symbol].dropna()
            if len(prices) < self.window:
                continue

            # Calculate z-score
            rolling_mean = prices.rolling(self.window).mean()
            rolling_std = prices.rolling(self.window).std()
            z_score = (prices - rolling_mean) / rolling_std

            # Generate signals
            signals.loc[z_score.index, symbol] = np.where(
                z_score < -self.threshold, 1.0,
                np.where(z_score > self.threshold, -1.0, 0.0),
            )

        return signals.fillna(0.0)

class PairsTradingStrategy(BaseStrategy):
    """Pairs trading strategy for two correlated assets.

    Trades the spread between two assets when it deviates from mean.
    """

    def __init__(self, pair: tuple, window: int = 30, threshold: float = 2.0, **kwargs) -> None:
        super().__init__("PairsTradingStrategy", pair=pair, window=window, threshold=threshold, **kwargs)
        self.pair = pair
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals."""
        if len(self.pair) != 2 or not all(symbol in data.columns for symbol in self.pair):
            msg = f"Both symbols in pair {self.pair} must be in data"
            raise ValueError(msg)

        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        symbol1, symbol2 = self.pair
        prices1 = data[symbol1].dropna()
        prices2 = data[symbol2].dropna()

        # Align data
        common_index = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_index]
        prices2 = prices2.loc[common_index]

        if len(common_index) < self.window:
            return signals

        # Calculate spread
        spread = prices1 - prices2
        rolling_mean = spread.rolling(self.window).mean()
        rolling_std = spread.rolling(self.window).std()
        z_score = (spread - rolling_mean) / rolling_std

        # Generate signals
        long_signal = z_score < -self.threshold  # Spread too negative, expect reversion
        short_signal = z_score > self.threshold   # Spread too positive, expect reversion

        signals.loc[common_index, symbol1] = np.where(
            long_signal, 1.0,
            np.where(short_signal, -1.0, 0.0),
        )
        signals.loc[common_index, symbol2] = np.where(
            long_signal, -1.0,
            np.where(short_signal, 1.0, 0.0),
        )

        return signals.fillna(0.0)

class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy.

    Enters positions when price breaks out of volatility bands.
    """

    def __init__(self, window: int = 20, multiplier: float = 2.0, **kwargs) -> None:
        super().__init__("VolatilityBreakoutStrategy", window=window, multiplier=multiplier, **kwargs)
        self.window = window
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            prices = data[symbol].dropna()
            if len(prices) < self.window:
                continue

            # Calculate Bollinger Bands
            rolling_mean = prices.rolling(self.window).mean()
            rolling_std = prices.rolling(self.window).std()
            upper_band = rolling_mean + (rolling_std * self.multiplier)
            lower_band = rolling_mean - (rolling_std * self.multiplier)

            # Generate signals
            signals.loc[prices.index, symbol] = np.where(
                prices > upper_band, 1.0,
                np.where(prices < lower_band, -1.0, 0.0),
            )

        return signals.fillna(0.0)
