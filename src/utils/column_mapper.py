"""
Column Mapping Utility for Analytics-STeeR

This module provides a unified approach to map CSV column names to 
standardized names that match the ontology and data model.

The ColumnMapper class handles the transformation of raw CSV column names
into standardized names that align with the ontology and data model.
It supports:
- Cleaning of column names (removing accents, standardizing case, etc.)
- Mapping to ontology property names
- Handling of special cases and variations
- Configuration through mapping files
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional


class ColumnMapper:
    """
    Maps CSV column names to standardized names based on the ontology.
    
    This class serves as the central component for transforming raw CSV column names
    into standardized names that match the ontology and data model. It handles:
    - Cleaning and normalization of column names
    - Mapping to ontology property names
    - Special case handling
    - Configuration management
    
    Attributes:
        ontology_path (str): Path to the ontology file in JSON-LD format
        mapping_config_path (str): Path to the column mapping configuration file
        ontology_properties (Dict[str, str]): Dictionary of property names from ontology
        custom_mapping (Dict[str, str]): User-defined column mappings
        known_prefixes (List[str]): List of common prefixes to handle
        accent_map (Dict[str, str]): Mapping of accented characters to their non-accented versions
    """
    
    def __init__(self, ontology_path: Optional[str] = None, mapping_config_path: Optional[str] = None):
        """
        Initialize the column mapper with paths to configuration files.
        
        Args:
            ontology_path: Path to the ontology file (JSON-LD format)
            mapping_config_path: Path to the mapping configuration file (JSON format)
            
        The constructor:
        1. Sets up paths to configuration files
        2. Loads ontology properties
        3. Loads custom mappings
        4. Initializes character mappings
        """
        self.ontology_path = ontology_path or str(Path(__file__).parent.parent.parent / "ontology" / "analytics_steer_ontology.jsonld")
        self.mapping_config_path = mapping_config_path or str(Path(__file__).parent / "column_mapping.json")
        
        # Load ontology and extract property names
        self.ontology_properties = self._extract_ontology_properties()
        
        # Load custom mapping if available
        self.custom_mapping = self._load_custom_mapping()
        
        # Common prefixes to handle in column names
        self.known_prefixes = ["eng_", "dnu_", "dnu-"]
        
        # Mapping of accented characters to their non-accented versions
        self.accent_map = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c',
            'É': 'e', 'È': 'e', 'Ê': 'e', 'Ë': 'e',
            'À': 'a', 'Â': 'a', 'Ä': 'a',
            'Î': 'i', 'Ï': 'i',
            'Ô': 'o', 'Ö': 'o',
            'Ù': 'u', 'Û': 'u', 'Ü': 'u',
            'Ç': 'c'
        }
    
    def _extract_ontology_properties(self) -> Dict[str, str]:
        """
        Extract properties from the ontology file.
        
        Returns:
            Dict[str, str]: Dictionary mapping property IDs to property names
            
        This method:
        1. Loads the ontology file
        2. Extracts properties from each class
        3. Creates mappings for both property IDs and names
        4. Handles errors gracefully
        """
        properties = {}
        
        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
                
            # Extract properties from each class in the ontology
            for class_name, class_def in ontology.get("classes", {}).items():
                for prop_name, prop_def in class_def.get("properties", {}).items():
                    # Store the property ID and the property name
                    prop_id = prop_def.get("@id", "").split(":")[-1]
                    properties[prop_id] = prop_name
                    properties[prop_name] = prop_name  # Also store the name itself
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading ontology: {e}")
        
        return properties
    
    def _load_custom_mapping(self) -> Dict[str, str]:
        """
        Load custom column mapping from configuration file.
        
        Returns:
            Dict[str, str]: Dictionary of custom column mappings
            
        This method:
        1. Attempts to load the mapping file
        2. Returns an empty dictionary if the file doesn't exist or is invalid
        3. Handles encoding and JSON parsing errors
        """
        try:
            with open(self.mapping_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If no mapping file or invalid, return empty mapping
            return {}
    
    def clean_column_name(self, column_name: str) -> str:
        """
        Clean a column name by standardizing format.
        
        Args:
            column_name: Original column name from CSV
            
        Returns:
            str: Cleaned and standardized column name
            
        This method:
        1. Converts to lowercase
        2. Replaces accented characters
        3. Handles special characters and spaces
        4. Removes duplicate underscores
        5. Handles known prefixes
        """
        if not column_name:
            return ""
        
        # Convert to lowercase
        result = column_name.lower()
        
        # Replace accented characters
        for char, replacement in self.accent_map.items():
            result = result.replace(char, replacement)
        
        # Replace spaces and special characters with underscores
        result = re.sub(r'[^a-z0-9]', '_', result)
        
        # Replace multiple underscores with a single one
        result = re.sub(r'_+', '_', result)
        
        # Remove leading/trailing underscores
        result = result.strip('_')
        
        # Handle known prefixes
        for prefix in self.known_prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):]
        
        return result
    
    def map_column(self, original_column: str) -> str:
        """
        Map an original column name to its standardized ontology property name.
        
        Args:
            original_column: Original column name from CSV
            
        Returns:
            str: Standardized column name matching ontology property
            
        This method follows a hierarchical mapping strategy:
        1. Checks for direct mapping in custom_mapping
        2. Cleans the column name
        3. Checks for exact match in ontology properties
        4. Checks for partial match in ontology properties
        5. Returns cleaned name as fallback
        """
        # Check for direct mapping
        if original_column in self.custom_mapping:
            return self.custom_mapping[original_column]
        
        # Clean the column name
        cleaned = self.clean_column_name(original_column)
        
        # Check if cleaned name matches ontology property
        if cleaned in self.ontology_properties:
            return self.ontology_properties[cleaned]
        
        # Try to match parts of the name to ontology properties
        parts = cleaned.split('_')
        for part in parts:
            if part in self.ontology_properties:
                return self.ontology_properties[part]
        
        # Return the cleaned name as fallback
        return cleaned
    
    def create_column_mapping(self, csv_columns: List[str]) -> Dict[str, str]:
        """
        Create a mapping dictionary from CSV columns to standardized names.
        
        Args:
            csv_columns: List of column names from CSV
            
        Returns:
            Dict[str, str]: Dictionary mapping original columns to standardized names
            
        This method:
        1. Processes each column in the input list
        2. Applies the mapping strategy to each column
        3. Returns a complete mapping dictionary
        """
        mapping = {}
        for column in csv_columns:
            mapping[column] = self.map_column(column)
        
        return mapping
    
    def generate_mapping_file(self, csv_columns: List[str], output_path: Optional[str] = None) -> None:
        """
        Generate a mapping configuration file from CSV columns.
        
        Args:
            csv_columns: List of column names from CSV
            output_path: Path to save the mapping file (defaults to self.mapping_config_path)
            
        This method:
        1. Creates a mapping for each column
        2. Saves the mapping to a JSON file
        3. Provides feedback on the operation
        """
        mapping = {}
        for column in csv_columns:
            cleaned = self.clean_column_name(column)
            mapping[column] = cleaned
        
        output_path = output_path or self.mapping_config_path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Mapping file generated at {output_path}")
    
    def save_mapping_config(self, mapping: Dict[str, str]) -> None:
        """
        Save a mapping configuration to file.
        
        Args:
            mapping: Dictionary mapping original columns to standardized names
            
        This method:
        1. Saves the provided mapping to the configuration file
        2. Uses JSON format with proper encoding
        3. Provides feedback on the operation
        """
        with open(self.mapping_config_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Mapping configuration saved to {self.mapping_config_path}")


# Dictionary of special case mappings for test-structure.csv columns
TEST_STRUCTURE_MAPPINGS = {
    # Client-related fields
    "SNAPSHOT_PERIOD": "snapshot_period",  # Period identifier (YYYYMM)
    "Client Group": "client_group_name",    # Name of the client's group
    "Direct Parent": "direct_parent_id",    # ID of the direct parent client
    "Client": "client_name",               # Name of the client
    "Client Id": "client_id",              # Unique identifier for the client
    "Industrie & Secteur": "industry_sector",  # Combined industry and sector
    "Industrie": "industry",               # Industry classification
    "Secteur": "sector",                   # Sector classification
    
    # Engagement-related fields
    "Master engagement": "master_engagement",  # Master engagement identifier
    "WBS L1 Id": "engagement_id",          # Work Breakdown Structure Level 1 ID
    "WBS L1": "engagement_name",           # Name of the engagement
    "WBS L2 Id": "wbs_l2_id",             # Work Breakdown Structure Level 2 ID
    "WBS L2": "wbs_l2_name",              # Name of the WBS Level 2
    
    # Role-related fields
    "Engagement Partner": "engagement_partner_name",  # Name of the engagement partner
    "Engagement Manager": "engagement_manager_name",  # Name of the engagement manager
    "Participating Partner": "participating_partner_name",  # Name of the participating partner
    "Participating Manager": "participating_manager_name",  # Name of the participating manager
    "Billing Partner": "billing_partner_name",  # Name of the billing partner
    "Billing Manager": "billing_manager_name",  # Name of the billing manager
    "Engagement Controller": "engagement_controller_name",  # Name of the engagement controller
    "Matricule LCSP": "lcsp_matricule",    # LCSP employee ID
    "LCSP": "lcsp_name",                   # Name of the LCSP
    
    # Business classification fields
    "Eng. Business Function": "business_function",  # Business function classification
    "Eng. Business Area": "business_area",  # Business area classification
    "Eng. Business Line": "business_line",  # Business line classification
    "Eng. Pôle": "pole",                   # Business pole classification
    "Eng. Business Sub Line": "business_sub_line",  # Business sub-line classification
    "Eng. Busines Sub Line Id": "business_sub_line_id",  # Business sub-line identifier
    "Eng. Région": "region",               # Geographic region
    
    # Status and currency fields
    "Statut WBS": "engagement_status",     # Engagement status
    "Devise de facturation": "billing_currency",  # Billing currency
    "Devise company code": "company_currency",  # Company currency
    "Active Flag": "is_active",            # Active status flag
    
    # Date fields
    "Date Début WBS": "start_date",        # Engagement start date
    "Date Fin WBS": "end_date",            # Engagement end date
    "Date clôture client": "client_closing_date",  # Client closing date
    "Mois clôture client": "client_closing_month",  # Client closing month
    "Fin mandat CAC": "cac_mandate_end",   # CAC mandate end date
    
    # Offering fields
    "Offering L5": "offering_l5_name",     # Level 5 offering name
    "Offering L6": "offering_l6_name",     # Level 6 offering name
    "Attest flag": "is_attested",          # Attestation status flag
    
    # Financial fields
    "% Discount WBS": "discount_percentage",  # Discount percentage
    "Net Revenus Annuel (carnet)": "net_revenue_carnet",  # Annual net revenue (carnet)
    "Net Revenue LTD (carnet)": "ltd_net_revenue_carnet",  # Lifetime net revenue (carnet)
    "% discount (carnet)": "carnet_discount_percentage",  # Carnet discount percentage
    "Gross Revenue Annuel (staffing)": "annual_gross_revenue_staffing",  # Annual gross revenue (staffing)
    "Gross Revenue YTG (staffing)": "ytg_gross_revenue_staffing",  # Year-to-go gross revenue (staffing)
    "Discount Annuel (staffing)": "annual_discount_staffing",  # Annual discount (staffing)
    "Net Revenue PTD (carnet)": "ptd_net_revenue_carnet",  # Period-to-date net revenue (carnet)
    "Discount manuel PTD": "ptd_manual_discount",  # Period-to-date manual discount
    "Date de mise à jour": "last_update_date"  # Last update date
}

def get_column_mapping(csv_columns: List[str], use_default: bool = True) -> Dict[str, str]:
    """
    Get column mapping for the provided CSV columns.
    
    Args:
        csv_columns: List of column names from CSV
        use_default: Whether to use default mappings for test-structure.csv
        
    Returns:
        Dict[str, str]: Dictionary mapping original columns to standardized names
        
    This function:
    1. Creates a ColumnMapper instance
    2. Uses default mappings if available and requested
    3. Falls back to generating new mappings if needed
    """
    mapper = ColumnMapper()
    
    if use_default and all(col in TEST_STRUCTURE_MAPPINGS for col in csv_columns):
        # If all columns are in the default mapping, use it
        return {col: TEST_STRUCTURE_MAPPINGS.get(col, mapper.clean_column_name(col)) 
                for col in csv_columns}
    
    # Otherwise, generate a new mapping
    return mapper.create_column_mapping(csv_columns)

def create_default_mapping_file():
    """
    Create the default mapping configuration file for test-structure.csv.
    
    This function:
    1. Creates a JSON file with the default mappings
    2. Uses UTF-8 encoding to handle special characters
    3. Provides feedback on the operation
    """
    output_path = Path(__file__).parent / "column_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(TEST_STRUCTURE_MAPPINGS, f, indent=2, ensure_ascii=False)
    print(f"Default mapping file created at {output_path}")


if __name__ == "__main__":
    # Create default mapping file when run directly
    create_default_mapping_file() 