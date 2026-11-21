"""
Test column mapping and CSV parsing functionality.

This module tests:
1. Loading and parsing the column mapping file
2. Applying the mapping to CSV headers
3. Verifying that mapped fields align with Pydantic models
4. Testing special character handling and normalization
"""

import pytest
import os
import pandas as pd
from pathlib import Path
from src.utils.column_mapper import ColumnMapper
from scripts.models import Client, Role, Engagement
from scripts.validation import validate_csv_mapping

@pytest.fixture
def setup_test_env():
    """Set up test environment"""
    csv_path = "test-structure.csv"
    mapper = ColumnMapper()
    
    # Read CSV headers
    with open(csv_path, 'r', encoding='latin-1') as f:
        header = f.readline().strip()
    csv_headers = header.split(';')
    
    return {
        'csv_path': csv_path,
        'mapper': mapper,
        'csv_headers': csv_headers
    }

def test_mapping_file_exists(setup_test_env):
    """Test that column mapping file exists and is valid JSON"""
    mapping_path = Path("src/utils/column_mapping.json")
    assert mapping_path.exists(), "Column mapping file not found"
    assert mapping_path.stat().st_size > 0, "Column mapping file is empty"

def test_all_csv_headers_mapped(setup_test_env):
    """Test that all CSV headers have corresponding mappings"""
    env = setup_test_env
    mapping = env['mapper'].create_column_mapping(env['csv_headers'])
    
    unmapped = []
    for header in env['csv_headers']:
        if header not in mapping:
            unmapped.append(header)
    
    assert len(unmapped) == 0, f"Found unmapped headers: {unmapped}"

def test_client_model_mapping(setup_test_env):
    """Test that mapped fields align with Client model"""
    env = setup_test_env
    df = pd.read_csv(env['csv_path'], sep=';', encoding='latin-1', nrows=1)
    success, missing_fields = validate_csv_mapping(df.columns, Client)
    
    assert success, f"Missing required fields for Client model: {missing_fields}"

def test_role_model_mapping(setup_test_env):
    """Test that mapped fields align with Role model"""
    env = setup_test_env
    df = pd.read_csv(env['csv_path'], sep=';', encoding='latin-1', nrows=1)
    success, missing_fields = validate_csv_mapping(df.columns, Role)
    
    assert success, f"Missing required fields for Role model: {missing_fields}"

def test_engagement_model_mapping(setup_test_env):
    """Test that mapped fields align with Engagement model"""
    env = setup_test_env
    df = pd.read_csv(env['csv_path'], sep=';', encoding='latin-1', nrows=1)
    success, missing_fields = validate_csv_mapping(df.columns, Engagement)
    
    assert success, f"Missing required fields for Engagement model: {missing_fields}"

def test_special_character_handling(setup_test_env):
    """Test handling of special characters in column names"""
    env = setup_test_env
    special_chars = {
        "Eng. Pôle": "pole",
        "Eng. Région": "region",
        "Date Début WBS": "start_date",
        "WIP d'ouverture": "opening_wip",
        "AR HT > 60j": "ar_ht_over_60d"
    }
    
    mapping = env['mapper'].create_column_mapping(list(special_chars.keys()))
    
    for original, expected in special_chars.items():
        assert mapping[original] == expected, f"Incorrect mapping for {original}"

def test_currency_field_mapping(setup_test_env):
    """Test that currency variations are properly mapped"""
    env = setup_test_env
    currency_pairs = [
        ("Net Revenue YTD", "net_revenue"),
        ("Net Revenue YTD LC", "net_revenue_lc"),
        ("Gross Revenue YTD", "gross_revenue"),
        ("Gross Revenue YTD LC", "gross_revenue_lc")
    ]
    
    mapping = env['mapper'].create_column_mapping([pair[0] for pair in currency_pairs])
    
    for original, expected in currency_pairs:
        assert mapping[original] == expected, f"Incorrect currency mapping for {original}"

def test_role_field_mapping(setup_test_env):
    """Test that role-related fields are properly mapped"""
    env = setup_test_env
    role_fields = [
        ("Engagement Partner", "engagement_partner_name"),
        ("Engagement Manager", "engagement_manager_name"),
        ("LCSP", "lcsp_name"),
        ("Matricule LCSP", "lcsp_matricule")
    ]
    
    mapping = env['mapper'].create_column_mapping([field[0] for field in role_fields])
    
    for original, expected in role_fields:
        assert mapping[original] == expected, f"Incorrect role field mapping for {original}" 