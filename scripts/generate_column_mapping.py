#!/usr/bin/env python3
"""
Generate column mapping file from CSV

This script reads the CSV file and generates a mapping file that can be used
to standardize column names according to the ontology.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.column_mapper import ColumnMapper

def generate_mapping_from_csv(csv_path, output_path=None):
    """
    Generate a column mapping file from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the mapping file (defaults to src/utils/column_mapping.json)
    """
    # Determine CSV delimiter
    with open(csv_path, 'r', encoding='latin-1') as f:
        sample = f.readline()
        delimiter = ';' if ';' in sample else ','
    
    # Read the header row only to get column names
    try:
        df = pd.read_csv(csv_path, delimiter=delimiter, nrows=0, encoding='latin-1')
        column_names = df.columns.tolist()
    except UnicodeDecodeError:
        # Try with UTF-8 if Latin-1 fails
        df = pd.read_csv(csv_path, delimiter=delimiter, nrows=0, encoding='utf-8')
        column_names = df.columns.tolist()
    
    print(f"Found {len(column_names)} columns in CSV file")
    print(f"First 5 columns: {column_names[:5]}")
    
    # Create column mapper and generate mapping file
    mapper = ColumnMapper()
    
    if not output_path:
        output_path = Path(__file__).parent.parent / "src" / "utils" / "column_mapping.json"
    
    mapper.generate_mapping_file(column_names, output_path)
    
    print(f"Generated mapping file with {len(column_names)} columns")
    print(f"Saved to {output_path}")
    
    # Print a sample of the mapping
    print("\nSample mappings:")
    sample_mappings = list(mapper.create_column_mapping(column_names).items())
    for original, cleaned in sample_mappings[:5]:
        print(f"  {original} -> {cleaned}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_column_mapping.py <csv_file> [output_file]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    generate_mapping_from_csv(csv_path, output_path) 