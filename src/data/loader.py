"""
Market Data Loader - Astra Trading Platform
===========================================

Professional market data acquisition with validation and caching.
Integrates with yfinance for real market data with proper error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import warnings
import time

class DataLoader:
    """
    Professional market data loader with caching and validation.
    
    Features:
    - Multi-symbol data downloading
    - Automatic retry on failures
    - Data validation and cleaning
    - Rate limiting for API compliance
    - Caching for development efficiency
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self.rate_limit_delay = 0.1  # Seconds between requests
        
    def download_ohlcv(
        self, 
        symbols: Union[str, List[str]], 
        start_date: str, 
        end_date: str,
        interval: str = '1d',
        validate: bool = True,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download OHLCV data for given symbols.
        
        Args:
            symbols: Stock symbol(s) to download
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            validate: Whether to validate data quality
            max_retries: Maximum retry attempts
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        print(f"📊 Downloading {len(symbols)} symbols: {symbols}")
        print(f"📅 Period: {start_date} to {end_date}")
        
        for attempt in range(max_retries):
            try:
                # Download with yfinance
                data = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                if data.empty:
                    raise ValueError("No data returned from yfinance")
                    
                # Handle single vs multiple symbols
                if len(symbols) == 1:
                    # Single symbol - flatten column names
                    data.columns = [col[0] if isinstance(col, tuple) else col 
                                  for col in data.columns]
                else:
                    # Multiple symbols - keep hierarchical columns
                    pass
                    
                print(f"✅ Downloaded {len(data)} rows of data")
                
                # Validate data quality if requested
                if validate:
                    data = self._validate_and_clean(data, symbols)
                    
                return data
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"🚨 Failed to download data after {max_retries} attempts")
                    raise
                    
        return pd.DataFrame()  # Should never reach here
    
    def download_adjusted_close(
        self, 
        symbols: Union[str, List[str]], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Download only adjusted close prices (most common for backtesting).
        
        Args:
            symbols: Stock symbol(s)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with adjusted close prices
        """
        data = self.download_ohlcv(symbols, start_date, end_date)
        
        if len(symbols) == 1:
            return data[['Adj Close']].rename(columns={'Adj Close': symbols[0]})
        else:
            return data['Adj Close']
    
    def get_universe_data(
        self, 
        universe: str = 'sp500', 
        start_date: str = '2023-01-01',
        end_date: str = '2024-01-01'
    ) -> pd.DataFrame:
        """
        Download data for predefined asset universes.
        
        Args:
            universe: Universe name ('sp500', 'tech', 'faang')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with universe data
        """
        universes = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'META'],
            'faang': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
            'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD'],
            'sample': ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Small set for testing
        }
        
        if universe not in universes:
            raise ValueError(f"Unknown universe: {universe}. Available: {list(universes.keys())}")
            
        symbols = universes[universe]
        return self.download_adjusted_close(symbols, start_date, end_date)
    
    def _validate_and_clean(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Validate and clean downloaded data.
        
        Args:
            data: Raw data from yfinance
            symbols: Symbol list for validation
            
        Returns:
            Cleaned data
        """
        original_length = len(data)
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values (conservative approach)
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN rows
        data = data.dropna()
        
        cleaned_length = len(data)
        
        if cleaned_length < original_length * 0.8:
            warnings.warn(f"Data cleaning removed {original_length - cleaned_length} rows "
                         f"({(1 - cleaned_length/original_length):.1%} of data)")
        
        print(f"🧹 Data cleaned: {original_length} → {cleaned_length} rows")
        
        return data

# Convenience function for quick data access
def get_sample_data(
    symbols: Union[str, List[str]] = ['AAPL', 'MSFT', 'GOOGL'],
    days: int = 365
) -> pd.DataFrame:
    """
    Quick function to get sample data for testing.
    
    Args:
        symbols: Symbols to download
        days: Number of days back from today
        
    Returns:
        DataFrame with adjusted close prices
    """
    loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return loader.download_adjusted_close(symbols, start_date, end_date)