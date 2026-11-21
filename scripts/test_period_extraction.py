#!/usr/bin/env python3
"""
Test script for parsing periods from test-structure.csv

This script demonstrates loading the CSV file and correctly parsing the 
SNAPSHOT_PERIOD column values into financial year and period components.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.column_mapper import get_column_mapping
from src.utils.period_utils import (
    parse_period, 
    get_period_label, 
    period_to_date_range, 
    extract_period_from_filename
)


def load_and_parse_periods(csv_path):
    """
    Load the CSV file and parse the periods from the SNAPSHOT_PERIOD column.
    
    Args:
        csv_path: Path to the CSV file
    """
    print(f"Loading CSV file: {csv_path}")
    
    # Check if the filename contains a period
    filename_period = extract_period_from_filename(csv_path)
    if filename_period:
        print(f"Detected period in filename: {filename_period} ({get_period_label(filename_period)})")
    
    # Determine CSV delimiter
    with open(csv_path, 'r', encoding='latin-1') as f:
        sample = f.readline()
        delimiter = ';' if ';' in sample else ','
    
    # Load CSV with pandas
    try:
        df = pd.read_csv(csv_path, delimiter=delimiter, encoding='latin-1')
    except UnicodeDecodeError:
        # Try with UTF-8 if Latin-1 fails
        df = pd.read_csv(csv_path, delimiter=delimiter, encoding='utf-8')
    
    # Get column names
    column_names = df.columns.tolist()
    print(f"Loaded {len(column_names)} columns: {column_names[:5]}...")
    
    # Map column names
    mapped_columns = get_column_mapping(column_names)
    standardized_names = {col: mapped_columns[col] for col in column_names}
    
    # Find the period column
    period_column = None
    for col, mapped in standardized_names.items():
        if mapped.lower() == 'snapshot_period':
            period_column = col
            break
    
    if not period_column:
        print("Error: Could not find SNAPSHOT_PERIOD column in CSV")
        return
    
    print(f"Found period column: '{period_column}' mapped to '{standardized_names[period_column]}'")
    
    # Get unique period values
    unique_periods = df[period_column].unique()
    print(f"Found {len(unique_periods)} unique period values:")
    
    # Parse and display periods
    print("\nParsed periods:")
    print("----------------------------------------------------------------------")
    print("| Period Value | Financial Year | Period | Label      | Date Range   |")
    print("----------------------------------------------------------------------")
    
    for period_val in sorted(unique_periods):
        financial_year, period = parse_period(period_val)
        label = get_period_label(period_val)
        
        # Get date range for the period
        start_date, end_date = period_to_date_range(period_val)
        date_range = f"{start_date} to {end_date}" if start_date and end_date else "Invalid"
        
        print(f"| {period_val:12} | {financial_year or 'Invalid':14} | {period or 'Invalid':6} | {label:11} | {date_range:12} |")
    
    print("----------------------------------------------------------------------")
    
    # Count by financial year
    fy_counts = {}
    for _, row in df.iterrows():
        period_val = row[period_column]
        fy, _ = parse_period(period_val)
        if fy:
            fy_counts[fy] = fy_counts.get(fy, 0) + 1
    
    if fy_counts:
        print("\nRows by Financial Year:")
        for fy, count in sorted(fy_counts.items()):
            print(f"FY {fy}: {count} rows")


if __name__ == "__main__":
    # Use provided CSV path or default to test-structure.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "test-structure.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    load_and_parse_periods(csv_path) 