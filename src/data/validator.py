"""
Data Validator - Astra Trading Platform
=======================================

Data quality validation and cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

class DataValidator:
    """
    Data quality validator for market data.
    
    Performs comprehensive checks on price data quality.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.validation_log = []
    
    def validate_price_data(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Comprehensive price data validation.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'date_range': (data.index.min(), data.index.max()) if not data.empty else None,
            'issues': []
        }
        
        is_valid = True
        
        # Check for empty data
        if data.empty:
            report['issues'].append('Data is empty')
            is_valid = False
            
        # Check for negative prices
        negative_prices = (data < 0).sum().sum()
        if negative_prices > 0:
            report['issues'].append(f'Found {negative_prices} negative prices')
            is_valid = False
            
        # Check for zero prices
        zero_prices = (data == 0).sum().sum()
        if zero_prices > 0:
            report['issues'].append(f'Found {zero_prices} zero prices')
            
        # Check for missing data
        missing_data = data.isnull().sum().sum()
        if missing_data > 0:
            report['issues'].append(f'Found {missing_data} missing values')
            
        # Check for extreme price movements (>50% daily)
        if not data.empty:
            returns = data.pct_change()
            extreme_moves = (abs(returns) > 0.5).sum().sum()
            if extreme_moves > 0:
                report['issues'].append(f'Found {extreme_moves} extreme price movements (>50%)')
                
        # Check data frequency consistency
        if len(data) > 1:
            date_diffs = pd.Series(data.index).diff().dropna()
            mode_diff = date_diffs.mode()[0] if not date_diffs.empty else None
            inconsistent_freq = (date_diffs != mode_diff).sum()
            if inconsistent_freq > len(data) * 0.1:  # More than 10% inconsistent
                report['issues'].append(f'Inconsistent data frequency detected')
                
        report['is_valid'] = is_valid
        
        if self.strict_mode and not is_valid:
            raise ValueError(f"Data validation failed: {report['issues']}")
            
        return is_valid, report
    
    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by fixing common issues.
        
        Args:
            data: Raw price data
            
        Returns:
            Cleaned price data
        """
        cleaned = data.copy()
        original_rows = len(cleaned)
        
        # Remove rows with all NaN values
        cleaned = cleaned.dropna(how='all')
        
        # Remove negative or zero prices
        cleaned = cleaned.where(cleaned > 0)
        
        # Forward fill missing values (conservative approach)
        cleaned = cleaned.fillna(method='ffill')
        
        # Remove any remaining NaN rows
        cleaned = cleaned.dropna()
        
        # Log cleaning results
        cleaned_rows = len(cleaned)
        if cleaned_rows < original_rows:
            self.validation_log.append(
                f"Data cleaning: {original_rows} → {cleaned_rows} rows "
                f"({(original_rows - cleaned_rows)/original_rows:.1%} removed)"
            )
            
        return cleaned