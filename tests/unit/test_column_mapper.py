"""
Unit tests for column mapping functionality
"""

import pytest
from src.utils.column_mapper import ColumnMapper
from src.models.load_data import load_raw_data

def test_csv_headers_exist(test_headers):
    """Test that CSV contains expected columns"""
    expected_columns = [
        'SNAPSHOT_PERIOD',
        'Client Group',
        'Client',
        'Client Id',
        'Industrie & Secteur'
    ]
    
    for col in expected_columns:
        assert col in test_headers, f"Expected column {col} not found in CSV"

def test_column_mapping(test_db, test_csv_path):
    """Test that column mapping is correctly applied"""
    # Load raw data to get normalized headers
    normalized_headers = load_raw_data(test_db, test_csv_path)
    
    # Verify key mappings
    expected_mappings = {
        'SNAPSHOT_PERIOD': 'snapshot_period',
        'Client Group': 'client_group',
        'Client': 'client',
        'Client Id': 'client_id',
        'Industrie & Secteur': 'industrie_and_secteur'
    }
    
    for orig, expected_norm in expected_mappings.items():
        result = test_db.execute("""
            SELECT normalized FROM header_map 
            WHERE original = ?
        """, [orig]).fetchone()
        
        assert result is not None, f"No mapping found for {orig}"
        assert result[0] == expected_norm, \
            f"Mapping for {orig} is {result[0]}, expected {expected_norm}"

def test_database_schema_sync(test_db, test_csv_path):
    """Test that mapped fields are in sync with database schema"""
    from src.models.schema_generator import SchemaGenerator
    
    # Create schema and load data
    schema_gen = SchemaGenerator(test_db)
    schema_gen.generate_schema()
    normalized_headers = load_raw_data(test_db, test_csv_path)
    
    # Test dim_clients required columns
    required_client_columns = ['client_id', 'client_name', 'client_group']
    for col in required_client_columns:
        assert col in normalized_headers, \
            f"Required column {col} for dim_clients not found in mapped headers"
    
    # Test fact_engagements required columns
    required_engagement_columns = ['engagement_id', 'snapshot_period', 'net_revenue_ytd']
    for col in required_engagement_columns:
        assert col in normalized_headers, \
            f"Required column {col} for fact_engagements not found in mapped headers" 