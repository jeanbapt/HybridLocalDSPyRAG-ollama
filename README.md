# Analytics-STeeR Project

This project provides a data pipeline for loading, validating, and analyzing structured data using DuckDB and Pydantic models, with ontology-based validation.

## Project Structure

```
.
├── data/                     # Database and data files
│   └── analytics_steer.duckdb  # Main DuckDB database
├── ontology/                 # Ontology definitions
│   └── analytics_steer_ontology.jsonld  # Ontology for Analytics-STeeR
├── scripts/                  # Python scripts
│   ├── __init__.py           # Package init
│   ├── db_manager.py         # Database operations
│   ├── load_data.py          # Data loading from CSV files
│   ├── load_test_data.py     # Specific script for test data loading
│   ├── models.py             # Pydantic data models
│   ├── run_analytics.py      # Main execution script
│   ├── schema_manager.py     # Database schema management
│   ├── test_data_loading.py  # Unit testing framework
│   └── validation.py         # Ontology-based validation
└── test-structure.csv        # Sample data CSV
```

## Requirements

- Python 3.8+
- DuckDB
- Pandas
- Pydantic

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Data Pipeline

```bash
python -m scripts.run_analytics
```

For specific options:

```bash
python -m scripts.run_analytics --help
```

Available options:
- `--csv-path`: Path to CSV file (default: `test-structure.csv`)
- `--db-path`: Path to DuckDB database (default: `data/analytics_steer.duckdb`)
- `--no-validation`: Skip validation tests
- `--clean`: Clean existing database before loading

### Running Unit Tests

```bash
PYTHONPATH=$PYTHONPATH:. python scripts/test_data_loading.py
```

## Key Features

### 1. Ontology-Based Data Validation

Data is validated against a formal ontology defined in JSON-LD format. This ensures:
- Data consistency
- Business rule enforcement
- Metadata-driven validation

### 2. Robust CSV Loading

Features include:
- Special character handling for column names (including French accents)
- Automatic standardization of column names
- European number format conversion (handles commas, periods, parentheses)
- Automatic mapping of CSV columns to model fields

### 3. Unified Column Mapping Strategy

The project implements a centralized column mapping strategy to ensure consistent handling of CSV column names:

- **Standardized Cleaning**: Converts column names to lowercase, removes accents, replaces spaces and special characters with underscores
- **Ontology Alignment**: Maps cleaned column names to ontology property names
- **Configuration-Based**: Uses a JSON configuration file to define mappings for known columns
- **Graceful Fallbacks**: When an exact mapping isn't found, falls back to intelligent matching strategies

Utility scripts include:
- `scripts/generate_column_mapping.py`: Generates mapping file from CSV headers
- `scripts/validate_model.py`: Validates mappings against the ontology

Example usage:
```bash
# Generate mapping from CSV file
python scripts/generate_column_mapping.py test-structure.csv

# Validate mapping against ontology
python scripts/validate_model.py
```

### 4. Schema Management

The database schema is defined using Pydantic models and includes:
- Strong typing
- Automated validation
- Default values
- Foreign key relationships
- Constraints matching business rules

### 5. Analytics Engine

Provides analytical capabilities:
- Period-over-period metrics
- Seasonality adjustments
- Role-based hierarchical view
- Authority relationship management

## Data Model Overview

### Core Dimensions
- **Time**: Snapshot periods, fiscal years, date ranges
- **Client**: Client information with hierarchical relationships
- **Person**: Person information with identification
- **Role**: Role assignments with authority relationships

### Facts and Metrics
- **Engagement**: Performance metrics for client engagements
- **PopMetrics**: Period-over-period growth metrics
- **SeasonalityMetrics**: Seasonality-adjusted performance metrics

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Make sure `PYTHONPATH` includes the project root directory
   - Check that all required `__init__.py` files exist

2. **CSV Format Issues**:
   - Check for BOM markers in CSV files
   - Verify separator is semicolon (`;`)
   - Confirm encoding (Latin-1 by default)

3. **Special Characters**:
   - Column names with accents or special characters are automatically cleaned
   - Numbers with European formatting (comma as decimal) are properly handled

4. **DuckDB Errors**:
   - DuckDB requires appropriate data types
   - Foreign key violations indicate data consistency issues
   - DuckDB settings may need adjustment based on version

## Future Enhancements

1. **Data Visualization Layer**: Integrate plotting and dashboarding capabilities
2. **Multiple Data Sources**: Extend to handle data from various sources
3. **Advanced Ontology Reasoning**: Implement inferencing based on ontology rules
4. **Incremental Loading**: Support for incremental data loading and updates
5. **Schema Evolution**: Handle changes in source data schema over time

## License

Proprietary - All rights reserved.

## Contact

For questions or assistance, please contact the Analytics-STeeR team. 