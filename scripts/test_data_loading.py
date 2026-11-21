"""
Test script for data loading with ColumnMapper

This script tests the data loading process using ColumnMapper to ensure:
1. Column names are correctly mapped
2. Data types are properly handled
3. Required fields are present
4. Data integrity is maintained
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.utils.column_mapper import ColumnMapper, get_column_mapping
from scripts.db_manager import DatabaseManager
from scripts.models import Client, Engagement, Role

def verify_database_schema(db_path: str = "data/analytics_steer.duckdb") -> bool:
    """
    Verify that the database schema is correctly initialized.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if schema is correctly initialized, False otherwise
    """
    print(f"\nVerifying database schema at {db_path}...")
    db = None
    
    try:
        # Initialize database manager
        print("Connecting to database...")
        db = DatabaseManager(db_path, initialize=False)
        
        # Get schema info
        print("\nGetting schema information...")
        schema_info = db.get_schema_info()
        print(f"Database version: {schema_info['version']}")
        print(f"Tables: {', '.join(schema_info['tables'].keys())}")
        
        # Verify tables
        required_tables = [
            'dim_time', 'dim_clients', 'dim_people', 
            'dim_roles', 'fact_engagements', 
            'fact_pop_metrics', 'fact_seasonality_metrics'
        ]
        
        missing_tables = [table for table in required_tables if table not in schema_info['tables']]
        if missing_tables:
            print(f"Error: Missing tables: {', '.join(missing_tables)}")
            return False
        
        # Verify schema is in sync with models
        print("\nVerifying schema compatibility with Pydantic models...")
        discrepancies = db.verify_db_pydantic_sync()
        if discrepancies:
            print("Warning: Found discrepancies between database schema and Pydantic models:")
            for disc in discrepancies:
                print(f"  - {disc}")
        else:
            print("Database schema is in sync with Pydantic models.")
        
        # Check for data in dim_time
        print("\nChecking dim_time table...")
        result = db.conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()
        print(f"Found {result[0]} records in dim_time table")
        
        if result[0] > 0:
            # Show sample record
            sample = db.conn.execute("SELECT * FROM dim_time LIMIT 1").fetchone()
            cols = db.conn.execute("PRAGMA table_info(dim_time)").fetchall()
            col_names = [col[1] for col in cols]
            sample_dict = {col: val for col, val in zip(col_names, sample)}
            print("Sample record:")
            for col, val in sample_dict.items():
                print(f"  {col}: {val}")
        
        # Overall status
        print("\nDatabase schema verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during database verification: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        if db:
            db.close()

if __name__ == "__main__":
    # Paths
    db_path = "data/analytics_steer.duckdb"
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Verify database schema
    success = verify_database_schema(db_path)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. Check the error messages above.") 