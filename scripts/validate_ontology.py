#!/usr/bin/env python
"""
Validates that the Pydantic models and the pruned ontology are aligned.
This script checks that:
1. All classes in the models match classes in the ontology
2. All fields in the models are represented in the ontology
3. Any fields in the ontology are accounted for in the models
"""

import json
import logging
import os
import sys
from typing import Dict, List, Set

from scripts.models import Client, Engagement, Person, PopMetrics, Role, SeasonalityMetrics, Time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Mapping from Pydantic model classes to ontology classes
MODEL_TO_ONTOLOGY_MAP = {
    Time: "Time",
    Client: "Client",
    Person: "Person",
    Role: "Role",
    Engagement: "Engagement",
    PopMetrics: "PopMetrics",
    SeasonalityMetrics: "SeasonalityMetrics"
}

def get_model_fields(model_classes: Dict) -> Dict[str, Set[str]]:
    """
    Extract field names from each Pydantic model
    """
    result = {}
    for model_class, ontology_class in model_classes.items():
        fields = set()
        for field_name in model_class.__annotations__:
            # Skip id field as it's used internally by Pydantic
            if field_name != "id":
                fields.add(field_name)
        result[ontology_class] = fields
    return result

def get_ontology_fields(ontology_file: str) -> Dict[str, Set[str]]:
    """
    Extract field names from the ontology
    """
    if not os.path.exists(ontology_file):
        logger.error(f"Ontology file not found: {ontology_file}")
        return {}

    with open(ontology_file, "r") as f:
        ontology = json.load(f)

    result = {}
    classes = ontology.get("classes", {})
    for class_name, class_info in classes.items():
        properties = class_info.get("properties", {})
        result[class_name] = set(properties.keys())

    return result

def validate_alignment(model_fields: Dict[str, Set[str]], ontology_fields: Dict[str, Set[str]]) -> bool:
    """
    Validate that model fields and ontology fields are aligned
    """
    all_valid = True
    
    for class_name, model_field_set in model_fields.items():
        if class_name not in ontology_fields:
            logger.error(f"Class {class_name} exists in models but not in ontology")
            all_valid = False
            continue
            
        onto_field_set = ontology_fields[class_name]
        
        # Check for fields in model but not in ontology
        missing_in_onto = model_field_set - onto_field_set
        if missing_in_onto:
            logger.error(f"Class {class_name} has fields in model but missing in ontology: {missing_in_onto}")
            all_valid = False
            
        # Check for fields in ontology but not in model
        missing_in_model = onto_field_set - model_field_set
        if missing_in_model:
            logger.error(f"Class {class_name} has fields in ontology but missing in model: {missing_in_model}")
            all_valid = False
            
        # Check for perfect match
        if model_field_set == onto_field_set:
            logger.info(f"✓ Class {class_name} fields match perfectly between model and ontology")
    
    return all_valid

def main():
    """
    Main function to validate ontology alignment with Pydantic models
    """
    ontology_file = "ontology/analytics_steer_ontology.jsonld"
    
    # Get fields from models and ontology
    model_fields = get_model_fields(MODEL_TO_ONTOLOGY_MAP)
    ontology_fields = get_ontology_fields(ontology_file)
    
    # Validate alignment
    is_valid = validate_alignment(model_fields, ontology_fields)
    
    if is_valid:
        logger.info("✓ Ontology is fully aligned with Pydantic models")
        return 0
    else:
        logger.error("✗ Ontology is not fully aligned with Pydantic models")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 