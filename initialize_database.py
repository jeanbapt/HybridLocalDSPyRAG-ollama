#!/usr/bin/env python3
"""
Initialize the Analytics-STeeR database and load test data.

This script initializes the database with the proper schema and optionally
loads test data from CSV files. It includes functionality to track schema versions
and clean up temporary files after initialization.
"""
import os
import sys
import json
import shutil
from pathlib import Path

# Add src to path for importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.db_manager import DatabaseManager
from src.models.load_data import load_csv_to_duckdb
from src.utils.column_mapper import get_column_mapping
from src.utils.period_utils import (
    extract_period_from_filename, 
    get_period_label,
    current_period
)
import pandas as pd

def detect_csv_encoding(file_path: str) -> str:
    """
    Detect the encoding of a CSV file by trying different encodings.
    Returns the first successful encoding.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Try to read a few lines
                for _ in range(5):
                    f.readline()
            return encoding
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not detect encoding for {file_path}")

def detect_csv_delimiter(file_path: str, encoding: str) -> str:
    """
    Detect the delimiter used in a CSV file.
    Returns the most likely delimiter.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        header = f.readline()
        
    # Check common delimiters
    delimiters = [';', ',', '\t', '|']
    counts = {d: header.count(d) for d in delimiters}
    
    # Return the delimiter with the highest count
    max_count = max(counts.values())
    if max_count > 0:
        return max(counts.items(), key=lambda x: x[1])[0]
    
    raise ValueError(f"Could not detect delimiter for {file_path}")

def backup_existing_files(files_to_backup):
    """
    Create backups of existing files before modifying them.
    
    Args:
        files_to_backup: List of file paths to backup
    """
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.bak"
            try:
                shutil.copy2(file_path, backup_path)
                print(f"Created backup of {file_path} at {backup_path}")
            except Exception as e:
                print(f"Warning: Could not backup {file_path}: {e}")

def cleanup_temp_files(keep_backups=True):
    """
    Clean up temporary files created during initialization.
    
    Args:
        keep_backups: Whether to keep backup files
    """
    # Temporary directories to clean
    temp_dirs = ["data/temp"]
    
    # Temporary files to remove
    temp_files = [
        "data/temp/mapping_cache.json",
        "data/temp/column_mapping_backup.json",
        "data/temp/schema_validation.log"
    ]
    
    # Process directories
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            try:
                # Only remove files, keep the directory structure
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path) and not file.endswith('.bak'):
                        os.remove(file_path)
                print(f"Cleaned temporary files in {dir_path}")
            except Exception as e:
                print(f"Warning: Could not clean {dir_path}: {e}")
    
    # Process individual files
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    # Remove backup files if requested
    if not keep_backups:
        backup_files = [f for f in os.listdir('.') if f.endswith('.bak')]
        for backup in backup_files:
            try:
                os.remove(backup)
                print(f"Removed backup file: {backup}")
            except Exception as e:
                print(f"Warning: Could not remove {backup}: {e}")

def main():
    """
    Initialize the database and load test data.
    
    Usage:
        python initialize_database.py [--skip-load] [--generate-mapping] [--period=YYYYPPP] [--force] [--no-cleanup]
    
    Options:
        --skip-load: Skip loading test data, only initialize the schema
        --generate-mapping: Generate column mapping file from CSV
        --period=YYYYPPP: Explicitly specify the period to use (e.g., --period=2025010)
        --force: Force reinitialization even if database already exists
        --no-cleanup: Skip cleanup of temporary files
    """
    # Default paths
    db_path = "data/analytics_steer.duckdb"
    csv_path = "test-structure.csv"
    
    # Parse command-line arguments
    skip_load = "--skip-load" in sys.argv
    generate_mapping = "--generate-mapping" in sys.argv
    force_init = "--force" in sys.argv
    no_cleanup = "--no-cleanup" in sys.argv
    
    # Check for explicit period argument
    period_arg = None
    for arg in sys.argv:
        if arg.startswith("--period="):
            period_arg = arg.split("=")[1]
            break
    
    # Try to detect period from filename if not explicitly provided
    if not period_arg and os.path.exists(csv_path):
        filename_period = extract_period_from_filename(csv_path)
        if filename_period:
            period_arg = filename_period
            print(f"Detected period from filename: {period_arg} ({get_period_label(period_arg)})")
    
    # If no period detected, use current period
    if not period_arg:
        period_arg = current_period()
        print(f"No period specified or detected. Using current period: {period_arg} ({get_period_label(period_arg)})")
    
    # Backup existing database if it exists and force is specified
    if force_init and os.path.exists(db_path):
        backup_existing_files([db_path])
        
        # Remove existing database to force reinitialization
        try:
            os.remove(db_path)
            print(f"Removed existing database at {db_path} for reinitialization")
        except Exception as e:
            print(f"Warning: Could not remove existing database: {e}")
    
    # Initialize the database schema
    print(f"Initializing database at {db_path}...")
    try:
        # Create database manager with initialization
        db_manager = DatabaseManager(db_path, initialize=True)
        
        # Verify schema is in sync with models
        discrepancies = db_manager.verify_db_pydantic_sync()
        if discrepancies:
            print("Warning: Found discrepancies between database schema and Pydantic models:")
            for disc in discrepancies:
                print(f"  - {disc}")
        else:
            print("Database schema is in sync with Pydantic models.")
        
        # Get schema info
        schema_info = db_manager.get_schema_info()
        print(f"Database initialized with version {schema_info['version']}")
        print(f"Created {len(schema_info['tables'])} tables")
        
        db_manager.close()
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return
    
    # Generate column mapping if requested
    if generate_mapping:
        print(f"Generating column mapping from {csv_path}...")
        
        try:
            # Backup existing mapping file if it exists
            mapping_file = "src/utils/column_mapping.json"
            backup_existing_files([mapping_file])
            
            # Detect encoding and delimiter
            encoding = detect_csv_encoding(csv_path)
            delimiter = detect_csv_delimiter(csv_path, encoding)
            print(f"Detected encoding: {encoding}, delimiter: {delimiter}")
            
            # Read CSV headers
            df = pd.read_csv(csv_path, delimiter=delimiter, nrows=0, encoding=encoding)
            column_names = df.columns.tolist()
            
            # Import this here to avoid circular imports
            from src.utils.column_mapper import ColumnMapper
            mapper = ColumnMapper()
            mapper.generate_mapping_file(column_names)
            
            print(f"Column mapping generated for {len(column_names)} columns")
        except Exception as e:
            print(f"Error generating column mapping: {e}")
    
    # Load test data if not skipped
    if not skip_load:
        if not os.path.exists(csv_path):
            print(f"Error: Test data file not found at {csv_path}")
            print("Skipping data load.")
            return
        
        print(f"Loading test data from {csv_path}...")
        try:
            # Detect encoding
            encoding = detect_csv_encoding(csv_path)
            print(f"Using detected encoding: {encoding}")
            
            # Load data with detected encoding
            load_csv_to_duckdb(csv_path, db_path, encoding=encoding, period=period_arg)
            print("Test data loaded successfully.")
        except Exception as e:
            print(f"Error loading test data: {e}")
            return
    
    # Clean up temporary files unless explicitly disabled
    if not no_cleanup:
        cleanup_temp_files(keep_backups=True)
    
    print("Database initialization complete!")

if __name__ == "__main__":
    main() 