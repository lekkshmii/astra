"""
Data Preprocessor - Astra Trading Platform
==========================================

Data preprocessing utilities for strategy development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class DataPreprocessor:
    """
    Data preprocessing utilities for market data.
    
    Handles feature engineering and data transformations.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to price data.
        
        Args:
            data: Price data (OHLCV or close prices)
            
        Returns:
            Data with additional technical indicator columns
        """
        result = data.copy()
        
        for column in data.columns:
            prices = data[column].dropna()
            if len(prices) < 50:  # Need minimum data
                continue
                
            # Simple moving averages
            result[f'{column}_SMA_10'] = prices.rolling(10).mean()
            result[f'{column}_SMA_20'] = prices.rolling(20).mean()
            result[f'{column}_SMA_50'] = prices.rolling(50).mean()
            
            # Exponential moving averages
            result[f'{column}_EMA_12'] = prices.ewm(span=12).mean()
            result[f'{column}_EMA_26'] = prices.ewm(span=26).mean()
            
            # Bollinger Bands
            sma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            result[f'{column}_BB_Upper'] = sma_20 + (2 * std_20)
            result[f'{column}_BB_Lower'] = sma_20 - (2 * std_20)
            
            # RSI
            result[f'{column}_RSI'] = self._calculate_rsi(prices)
            
            # MACD
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            result[f'{column}_MACD'] = ema_12 - ema_26
            result[f'{column}_MACD_Signal'] = result[f'{column}_MACD'].ewm(span=9).mean()
            
        return result
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def normalize_prices(self, data: pd.DataFrame, method: str = 'returns') -> pd.DataFrame:
        """
        Normalize price data for analysis.
        
        Args:
            data: Price data
            method: 'returns', 'log_returns', or 'standardize'
            
        Returns:
            Normalized data
        """
        if method == 'returns':
            return data.pct_change().dropna()
        elif method == 'log_returns':
            return np.log(data / data.shift(1)).dropna()
        elif method == 'standardize':
            return (data - data.mean()) / data.std()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def create_lagged_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged versions of features.
        
        Args:
            data: Input data
            lags: List of lag periods to create
            
        Returns:
            Data with lagged features
        """
        result = data.copy()
        
        for column in data.columns:
            for lag in lags:
                result[f'{column}_lag_{lag}'] = data[column].shift(lag)
                
        return result.dropna()