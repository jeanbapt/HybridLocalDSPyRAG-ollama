"""
Shared pytest fixtures for Analytics-STeeR tests
"""

import os
import pytest
import duckdb
from pathlib import Path

def get_data_dir():
    """Get the data directory path, creating it if it doesn't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture(scope="function")
def test_db():
    """Fixture providing a test database connection"""
    # Setup - ensure database is always in data directory
    data_dir = get_data_dir()
    test_db_path = data_dir / "test_analytics_steer.duckdb"
    
    # Remove existing test database if any
    if test_db_path.exists():
        test_db_path.unlink()
    
    # Create connection with proper temp directory in data/temp
    temp_dir = data_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    con = duckdb.connect(str(test_db_path))
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    
    yield con
    
    # Teardown
    con.close()
    if test_db_path.exists():
        test_db_path.unlink()

@pytest.fixture(scope="session")
def test_csv_path():
    """Fixture providing path to test CSV file"""
    # First check in data directory
    data_dir = get_data_dir()
    csv_path = data_dir / "test-structure.csv"
    
    # If not in data dir, check in root (for backward compatibility)
    if not csv_path.exists():
        csv_path = Path("test-structure.csv")
        if not csv_path.exists():
            pytest.skip("test-structure.csv not found in data/ or root directory")
    
    return str(csv_path)

@pytest.fixture(scope="function")
def test_headers(test_csv_path):
    """Fixture providing original headers from test CSV"""
    import pandas as pd
    df = pd.read_csv(test_csv_path, sep=';', encoding='latin-1', nrows=0)
    return df.columns.tolist() 