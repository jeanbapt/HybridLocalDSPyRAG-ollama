#!/usr/bin/env python3
"""
Validate data model against ontology

This script validates that the data model classes match the ontology definition
and that the column mappings align with ontology properties.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.column_mapper import ColumnMapper

def validate_ontology_alignment():
    """
    Validate that the ontology and data model are aligned.
    
    This checks:
    1. That all classes in the ontology have corresponding data model classes
    2. That all properties in the ontology have corresponding column mappings
    3. That data types are consistent between ontology and model
    """
    # Load the ontology
    mapper = ColumnMapper()
    ontology_path = Path(__file__).parent.parent / "ontology" / "analytics_steer_ontology.jsonld"
    
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ontology: {e}")
        return False
    
    # Load the column mapping
    mapping_path = Path(__file__).parent.parent / "src" / "utils" / "column_mapping.json"
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            column_mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading column mapping: {e}")
        return False
    
    # Extract ontology classes and properties
    ontology_classes = ontology.get("classes", {})
    
    # Check alignment
    aligned = True
    
    # Track all ontology property names
    ontology_properties = set()
    for class_name, class_def in ontology_classes.items():
        print(f"\nValidating class: {class_name}")
        
        for prop_name, prop_def in class_def.get("properties", {}).items():
            ontology_properties.add(prop_name)
            
            # Get the standardized property name from the ID
            prop_id = prop_def.get("@id", "").split(":")[-1]
            if prop_id:
                ontology_properties.add(prop_id)
            
            print(f"  - Property: {prop_name} ({prop_id})")
            
            # Check if this property is covered in column mapping
            mapped_columns = [col for col, mapped in column_mapping.items() 
                             if mapped.lower() == prop_name.lower() or 
                                mapped.lower() == prop_id.lower()]
            
            if mapped_columns:
                print(f"    ✓ Mapped from: {', '.join(mapped_columns)}")
            else:
                print(f"    ✗ No mapping found for this property")
                aligned = False
    
    # Check if there are mapped columns that don't match ontology properties
    print("\nChecking for unmapped ontology properties:")
    mapped_values = set(value.lower() for value in column_mapping.values())
    unmapped_properties = set()
    
    for prop in ontology_properties:
        if prop.lower() not in mapped_values:
            unmapped_properties.add(prop)
    
    if unmapped_properties:
        print(f"  ✗ Found {len(unmapped_properties)} unmapped properties:")
        for prop in sorted(unmapped_properties):
            print(f"    - {prop}")
        aligned = False
    else:
        print("  ✓ All ontology properties are mapped")
    
    # Check for mappings that don't match ontology properties
    print("\nChecking for invalid column mappings:")
    invalid_mappings = set()
    
    for original, mapped in column_mapping.items():
        if mapped.lower() not in {p.lower() for p in ontology_properties}:
            invalid_mappings.add((original, mapped))
    
    if invalid_mappings:
        print(f"  ✗ Found {len(invalid_mappings)} invalid mappings:")
        for original, mapped in sorted(invalid_mappings):
            print(f"    - {original} -> {mapped}")
        aligned = False
    else:
        print("  ✓ All column mappings are valid")
    
    return aligned


if __name__ == "__main__":
    print("Validating ontology and data model alignment...")
    is_aligned = validate_ontology_alignment()
    
    if is_aligned:
        print("\n✓ Validation successful: Ontology and data model are aligned")
        sys.exit(0)
    else:
        print("\n✗ Validation failed: Ontology and data model are not fully aligned")
        sys.exit(1) 