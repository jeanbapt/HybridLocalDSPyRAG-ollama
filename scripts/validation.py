"""
Ontology-based validation module for Analytics-STeeR.

This module provides functionality to validate data against the defined ontology,
ensuring consistency with business rules and data integrity constraints.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OntologyValidator:
    """Validates data against the Analytics-STeeR ontology."""
    
    def __init__(self, ontology_path: Optional[str] = None):
        """
        Initialize the ontology validator.
        
        Args:
            ontology_path: Path to the ontology file (JSON-LD format)
        """
        self.ontology_path = ontology_path or Path("ontology/analytics_steer_ontology.jsonld")
        self.ontology = self._load_ontology()
        self.validation_rules = self._extract_validation_rules()
        
    def _load_ontology(self) -> Dict[str, Any]:
        """Load the ontology from the JSON-LD file."""
        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Ontology file not found at {self.ontology_path}. Using empty ontology.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in ontology file {self.ontology_path}")
            return {}
    
    def _extract_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Extract validation rules from the ontology."""
        rules = {}
        
        if not self.ontology:
            return rules
            
        # Process classes in the ontology
        for cls in self.ontology.get("@graph", []):
            if cls.get("@type") == "owl:Class":
                class_id = cls.get("@id", "").split(":")[-1]
                if not class_id:
                    continue
                    
                rules[class_id] = {
                    "properties": {},
                    "constraints": []
                }
                
                # Process properties
                for prop in cls.get("properties", []):
                    prop_name = prop.get("name")
                    if not prop_name:
                        continue
                        
                    rules[class_id]["properties"][prop_name] = {
                        "datatype": prop.get("range", "xsd:string").split(":")[-1],
                        "required": prop.get("required", False),
                        "unique": prop.get("unique", False),
                        "min": prop.get("minValue"),
                        "max": prop.get("maxValue"),
                        "pattern": prop.get("pattern")
                    }
                
                # Process constraints
                for constraint in cls.get("constraints", []):
                    rules[class_id]["constraints"].append({
                        "type": constraint.get("type"),
                        "expression": constraint.get("expression"),
                        "message": constraint.get("message")
                    })
        
        return rules
    
    def validate_model(self, model: Type[BaseModel], data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against a Pydantic model and ontology rules.
        
        Args:
            model: Pydantic model class
            data: Data to validate
            
        Returns:
            Tuple of (success, error_messages)
        """
        errors = []
        
        # First, validate against Pydantic model
        try:
            model_instance = model(**data)
        except ValidationError as e:
            for error in e.errors():
                errors.append(f"Validation error: {error['loc'][0]} - {error['msg']}")
            return False, errors
        
        # Then validate against ontology rules
        model_name = model.__name__
        if model_name not in self.validation_rules:
            logger.warning(f"No ontology rules found for model {model_name}")
            return True, []
        
        # Validate property constraints from ontology
        model_rules = self.validation_rules[model_name]
        for prop_name, prop_rules in model_rules["properties"].items():
            value = getattr(model_instance, prop_name, None)
            
            # Skip if not required and None
            if value is None and not prop_rules.get("required", False):
                continue
                
            # Validate min/max
            if prop_rules.get("min") is not None and value < prop_rules["min"]:
                errors.append(f"{prop_name} must be at least {prop_rules['min']}")
            
            if prop_rules.get("max") is not None and value > prop_rules["max"]:
                errors.append(f"{prop_name} must be at most {prop_rules['max']}")
        
        # Validate custom constraints
        for constraint in model_rules["constraints"]:
            constraint_type = constraint.get("type")
            expression = constraint.get("expression")
            
            if not constraint_type or not expression:
                continue
                
            if constraint_type == "SPARQL":
                # TODO: Implement SPARQL constraint validation
                pass
            elif constraint_type == "Python":
                try:
                    # Convert attribute references in expression
                    for attr_name in dir(model_instance):
                        if attr_name.startswith('_'):
                            continue
                        token = f"${attr_name}"
                        if token in expression:
                            value = getattr(model_instance, attr_name)
                            if isinstance(value, str):
                                expression = expression.replace(token, f"'{value}'")
                            else:
                                expression = expression.replace(token, str(value))
                    
                    result = eval(expression)
                    if not result:
                        errors.append(constraint.get("message", f"Constraint failed: {expression}"))
                except Exception as e:
                    logger.error(f"Error evaluating constraint {expression}: {e}")
                    errors.append(f"Error in constraint: {e}")
        
        return len(errors) == 0, errors
    
    def validate_data_set(self, model: Type[BaseModel], data_set: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Validate a dataset against a model and ontology rules.
        
        Args:
            model: Pydantic model class
            data_set: List of data dictionaries
            
        Returns:
            Dictionary mapping row indices to error messages
        """
        validation_errors = {}
        
        # Track unique values for uniqueness constraints
        unique_values: Dict[str, Set[Any]] = {}
        model_name = model.__name__
        
        # Initialize unique tracking for properties with uniqueness constraint
        if model_name in self.validation_rules:
            for prop_name, prop_rules in self.validation_rules[model_name]["properties"].items():
                if prop_rules.get("unique", False):
                    unique_values[prop_name] = set()
        
        # Validate each row
        for idx, row_data in enumerate(data_set):
            # Validate against model and ontology
            success, errors = self.validate_model(model, row_data)
            
            # Check uniqueness constraints
            if model_name in self.validation_rules:
                for prop_name, prop_rules in self.validation_rules[model_name]["properties"].items():
                    if prop_rules.get("unique", False) and prop_name in row_data:
                        value = row_data[prop_name]
                        if value in unique_values[prop_name]:
                            errors.append(f"{prop_name} must be unique (duplicate value: {value})")
                        else:
                            unique_values[prop_name].add(value)
            
            if not success or errors:
                validation_errors[idx] = errors
        
        return validation_errors
    
    def get_model_constraints(self, model_name: str) -> Dict[str, Any]:
        """
        Get constraints for a specific model from the ontology.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of constraints
        """
        return self.validation_rules.get(model_name, {"properties": {}, "constraints": []})
    
    def generate_validation_report(self, validation_errors: Dict[int, List[str]]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_errors: Dictionary mapping row indices to error messages
            
        Returns:
            Formatted report string
        """
        if not validation_errors:
            return "Validation successful: No errors found."
        
        report = ["Validation failed with the following errors:"]
        
        for idx, errors in validation_errors.items():
            report.append(f"\nRow {idx + 1}:")
            for error in errors:
                report.append(f"  - {error}")
        
        return "\n".join(report)


def validate_csv_mapping(csv_columns: List[str], model: Type[BaseModel]) -> Tuple[bool, List[str]]:
    """
    Validate that a CSV has all required columns for a model.
    
    Args:
        csv_columns: List of column names from CSV
        model: Pydantic model class
        
    Returns:
        Tuple of (success, missing_required_fields)
    """
    # Get all required fields from the model
    required_fields = []
    for field_name, field in model.model_fields.items():
        if field.is_required:
            required_fields.append(field_name)
    
    # Clean CSV column names to standardized format for comparison
    cleaned_columns = [clean_column_name(col) for col in csv_columns]
    
    # Check if all required fields are present
    missing_fields = []
    for field in required_fields:
        if field.lower() not in [col.lower() for col in cleaned_columns]:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def clean_column_name(name: str) -> str:
    """
    Clean a column name by removing special characters and standardizing format.
    
    Args:
        name: Original column name
        
    Returns:
        Cleaned column name
    """
    import re
    
    # Standardize accented characters
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c'
    }
    
    # Convert to lowercase
    result = name.lower()
    
    # Replace accented characters
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    # Replace special characters with underscores
    result = re.sub(r'[^a-z0-9]', '_', result)
    
    # Replace multiple underscores with a single one
    result = re.sub(r'_+', '_', result)
    
    # Remove leading/trailing underscores
    result = result.strip('_')
    
    return result 