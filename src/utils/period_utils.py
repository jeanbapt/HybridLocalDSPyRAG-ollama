"""
Period utilities for Analytics-STeeR

This module provides utilities for working with financial periods.
"""

from typing import Tuple, Optional, Union
from datetime import datetime, date


def parse_period(period_str: Union[str, int]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a period string/integer into financial year and period components.
    
    Format: YYYYPPP where YYYY is the financial year and PPP is the period
    Example: 2025010 represents FY2025, Period 10
    
    Args:
        period_str: Period string in format 'YYYYPPP' (e.g., '2025010', '2024002')
        
    Returns:
        Tuple of (financial_year, period)
    """
    if not period_str:
        return None, None
    
    # Convert to string if needed
    period_str = str(period_str)
    
    # Validate format
    if not period_str.isdigit() or len(period_str) < 7:
        return None, None
    
    financial_year = int(period_str[:4])
    period = int(period_str[4:])
    
    return financial_year, period


def format_period(financial_year: int, period: int) -> str:
    """
    Format financial year and period into a standard period string.
    
    Args:
        financial_year: The financial year (e.g., 2025)
        period: The period number (e.g., 10)
        
    Returns:
        Period string in format 'YYYYPPP'
    """
    return f"{financial_year}{period:03d}"


def get_period_label(period_str: Union[str, int]) -> str:
    """
    Get a human-readable label for a period.
    
    Args:
        period_str: Period string in format 'YYYYPPP'
        
    Returns:
        Human-readable label (e.g., "FY2025-P10")
    """
    financial_year, period = parse_period(period_str)
    if not financial_year or not period:
        return "Invalid Period"
    
    return f"FY{financial_year}-P{period}"


def period_to_date_range(period_str: Union[str, int]) -> Tuple[Optional[date], Optional[date]]:
    """
    Convert a period to start and end dates.
    
    This assumes a standard financial year starting in January.
    For a different financial year start, adjustments would be needed.
    
    Args:
        period_str: Period string in format 'YYYYPPP'
        
    Returns:
        Tuple of (start_date, end_date)
    """
    financial_year, period = parse_period(period_str)
    if not financial_year or not period:
        return None, None
    
    # Assuming monthly periods starting from January
    month = period
    
    # Calculate start date (first day of the month)
    if 1 <= month <= 12:
        start_date = date(financial_year, month, 1)
    else:
        return None, None
    
    # Calculate end date (last day of the month)
    if month == 12:
        end_date = date(financial_year + 1, 1, 1)
    else:
        end_date = date(financial_year, month + 1, 1)
    
    end_date = date(end_date.year, end_date.month, 1)
    
    # Adjust to get the last day of the month
    from datetime import timedelta
    end_date = end_date - timedelta(days=1)
    
    return start_date, end_date


def current_period() -> str:
    """
    Get the current period based on today's date.
    
    Returns:
        Current period in format 'YYYYPPP'
    """
    today = datetime.now()
    return format_period(today.year, today.month)


def extract_period_from_filename(filename: str) -> Optional[str]:
    """
    Extract period from a filename if it matches the pattern.
    
    Args:
        filename: Filename possibly containing a period
        
    Returns:
        Period string if found, None otherwise
    """
    import re
    import os
    
    # Get just the filename without path
    basename = os.path.basename(filename)
    
    # Look for 7-digit pattern that could be a period
    match = re.search(r'\b(\d{7})\b', basename)
    if match:
        period_str = match.group(1)
        financial_year, period = parse_period(period_str)
        
        # Validate that it looks like a reasonable period
        if financial_year and 2000 <= financial_year <= 2100 and 1 <= period <= 999:
            return period_str
    
    return None 