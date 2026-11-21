"""
Script to load test data from test-structure.csv into Analytics-STeeR database.

This script cleans column names, maps them to database fields, and loads the data
into the appropriate tables with validation against the ontology.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import pandas as pd
import logging
from datetime import datetime
from scripts.db_manager import DatabaseManager
from scripts.models import (
    Time, Client, Person, Role, Engagement,
    PopMetrics, SeasonalityMetrics
)
from scripts.validation import OntologyValidator
from utils.column_mapper import ColumnMapper

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(csv_path: str = "test-structure.csv", db_path: str = "data/analytics_steer.duckdb"):
    """
    Load data from test-structure.csv into Analytics-STeeR database.
    
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the DuckDB database
    """
    logger.info(f"Loading data from {csv_path} into {db_path}")
    
    # Initialize database connection
    db_manager = DatabaseManager(db_path)
    
    # Initialize ontology validator
    validator = OntologyValidator()
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Initialize column mapper
        mapper = ColumnMapper()
        
        # Create column mapping
        column_mapping = mapper.create_column_mapping(df.columns.tolist())
        
        # Rename columns using mapping
        df.rename(columns=column_mapping, inplace=True)
        
        # Process and load data for each dimension and fact table
        load_time_dimension(df, db_manager, validator)
        load_client_dimension(df, db_manager, validator)
        load_person_dimension(df, db_manager, validator)
        load_role_dimension(df, db_manager, validator)
        load_engagement_facts(df, db_manager, validator)
        
        # Calculate derived metrics
        calculate_metrics(db_manager)
        
        logger.info("Data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise
    finally:
        # Close database connection
        db_manager.close()

def load_time_dimension(df, db_manager, validator):
    """Load time dimension from DataFrame."""
    logger.info("Loading time dimension")
    
    # Get unique snapshot periods
    unique_periods = df['snapshot_period'].unique()
    
    time_records = []
    errors = []
    
    for period in unique_periods:
        # Extract year and period (assuming format YYYYMM)
        if isinstance(period, str) and len(period) >= 6:
            year = int(period[:4])
            month = int(period[4:6])
            
            # Create fiscal year (assuming fiscal year starts in April)
            fiscal_year_start_month = 4
            fiscal_year = year if month >= fiscal_year_start_month else year - 1
            fiscal_period = month - fiscal_year_start_month + 1
            if fiscal_period <= 0:
                fiscal_period += 12
            
            # Calculate period start and end dates
            debut_periode = f"{year}-{month:02d}-01"
            # Simplified end date calculation (not accounting for different month lengths)
            if month == 12:
                fin_periode = f"{year+1}-01-01"
            else:
                fin_periode = f"{year}-{month+1:02d}-01"
            
            # Create time record
            time_data = {
                'snapshot_period': period,
                'year': year,
                'period': month,
                'period_type': 'MONTH',
                'fiscal_year': fiscal_year,
                'fiscal_period': fiscal_period,
                'debut_periode': debut_periode,
                'fin_periode': fin_periode
            }
            
            # Validate against ontology
            success, validation_errors = validator.validate_model(Time, time_data)
            if not success:
                errors.extend([f"Error in period {period}: {err}" for err in validation_errors])
                continue
                
            # Create Pydantic model
            time_model = Time(**time_data)
            time_records.append(time_model.dict())
        else:
            errors.append(f"Invalid snapshot_period format: {period}")
    
    # Raise error if validation failed
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Time dimension validation failed:\n{error_msg}")
    
    # Insert records into database
    if time_records:
        db_manager.insert_time_dimension(time_records)
        logger.info(f"Inserted {len(time_records)} time dimension records")

def load_client_dimension(df, db_manager, validator):
    """Load client dimension from DataFrame."""
    logger.info("Loading client dimension")
    
    # Get relevant columns for client dimension
    client_cols = ['client_id', 'client_name', 'client_group', 'direct_parent', 'industry', 'region']
    client_cols = [col for col in client_cols if col in df.columns]
    
    # Get unique client records
    if 'client_id' in df.columns and 'client_name' in df.columns:
        # Rename client_group to client_group_name if it exists
        if 'client_group' in df.columns:
            df = df.rename(columns={'client_group': 'client_group_name'})
            client_cols = [col if col != 'client_group' else 'client_group_name' for col in client_cols]
            
        client_df = df[client_cols].drop_duplicates(subset=['client_id'])
        
        client_records = []
        errors = []
        
        for _, row in client_df.iterrows():
            client_data = row.to_dict()
            
            # Handle parent relationship
            if 'direct_parent' in client_data and pd.isna(client_data['direct_parent']):
                client_data['direct_parent'] = None
            
            # Validate against ontology
            success, validation_errors = validator.validate_model(Client, client_data)
            if not success:
                errors.extend([f"Error in client {client_data.get('client_id')}: {err}" for err in validation_errors])
                continue
                
            # Create Pydantic model
            client_model = Client(**client_data)
            client_records.append(client_model.dict())
        
        # Raise error if validation failed
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Client dimension validation failed:\n{error_msg}")
        
        # Insert records into database
        if client_records:
            db_manager.insert_client_dimension(client_records)
            logger.info(f"Inserted {len(client_records)} client dimension records")
    else:
        logger.warning("Required client columns missing, skipping client dimension")

def load_person_dimension(df, db_manager, validator):
    """Load person dimension from DataFrame."""
    logger.info("Loading person dimension")
    
    # Get relevant columns for person dimension
    person_cols = ['person_id', 'matricule', 'full_name', 'first_name', 'last_name']
    person_cols = [col for col in person_cols if col in df.columns]
    
    # Get unique person records
    if 'person_id' in df.columns and 'full_name' in df.columns:
        person_df = df[person_cols].drop_duplicates(subset=['person_id'])
        
        person_records = []
        errors = []
        
        for _, row in person_df.iterrows():
            person_data = row.to_dict()
            
            # Validate against ontology
            success, validation_errors = validator.validate_model(Person, person_data)
            if not success:
                errors.extend([f"Error in person {person_data.get('person_id')}: {err}" for err in validation_errors])
                continue
                
            # Create Pydantic model
            person_model = Person(**person_data)
            person_records.append(person_model.dict())
        
        # Raise error if validation failed
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Person dimension validation failed:\n{error_msg}")
        
        # Insert records into database
        if person_records:
            db_manager.insert_person_dimension(person_records)
            logger.info(f"Inserted {len(person_records)} person dimension records")
    else:
        logger.warning("Required person columns missing, skipping person dimension")

def load_role_dimension(df, db_manager, validator):
    """Load role dimension from DataFrame."""
    logger.info("Loading role dimension")
    
    # Get relevant columns for role dimension
    role_cols = ['role_id', 'person_id', 'client_id', 'role_type', 'authority_level', 
                'start_date', 'end_date', 'supervisor_role_id']
    role_cols = [col for col in role_cols if col in df.columns]
    
    # Get unique role records
    if all(col in df.columns for col in ['role_id', 'person_id', 'client_id', 'role_type']):
        role_df = df[role_cols].drop_duplicates(subset=['role_id'])
        
        role_records = []
        errors = []
        
        for _, row in role_df.iterrows():
            role_data = row.to_dict()
            
            # Handle dates
            for date_field in ['start_date', 'end_date']:
                if date_field in role_data and pd.isna(role_data[date_field]):
                    role_data[date_field] = None
            
            # Handle supervisor relationship
            if 'supervisor_role_id' in role_data and pd.isna(role_data['supervisor_role_id']):
                role_data['supervisor_role_id'] = None
            
            # Validate against ontology
            success, validation_errors = validator.validate_model(Role, role_data)
            if not success:
                errors.extend([f"Error in role {role_data.get('role_id')}: {err}" for err in validation_errors])
                continue
                
            # Create Pydantic model
            role_model = Role(**role_data)
            role_records.append(role_model.dict())
        
        # Raise error if validation failed
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Role dimension validation failed:\n{error_msg}")
        
        # Insert records into database
        if role_records:
            db_manager.insert_role_dimension(role_records)
            logger.info(f"Inserted {len(role_records)} role dimension records")
    else:
        logger.warning("Required role columns missing, skipping role dimension")

def load_engagement_facts(df, db_manager, validator):
    """Load engagement facts from DataFrame."""
    logger.info("Loading engagement facts")
    
    # Get relevant columns for engagement facts
    engagement_cols = [
        'engagement_id', 'snapshot_period', 'client_id', 'lcsp_role_id', 
        'engagement_partner_role_id', 'engagement_manager_role_id',
        'net_revenue', 'total_costs', 'net_margin', 'margin_percentage'
    ]
    engagement_cols = [col for col in engagement_cols if col in df.columns]
    
    # Get engagement records
    if all(col in df.columns for col in ['engagement_id', 'snapshot_period', 'client_id']):
        engagement_df = df[engagement_cols].drop_duplicates(subset=['engagement_id', 'snapshot_period'])
        
        engagement_records = []
        errors = []
        
        for _, row in engagement_df.iterrows():
            engagement_data = row.to_dict()
            
            # Convert numeric fields to appropriate type
            numeric_fields = ['net_revenue', 'total_costs', 'net_margin', 'margin_percentage']
            for field in numeric_fields:
                if field in engagement_data:
                    try:
                        if pd.isna(engagement_data[field]):
                            engagement_data[field] = None
                        else:
                            engagement_data[field] = float(engagement_data[field])
                    except (ValueError, TypeError):
                        errors.append(f"Error converting {field} to number: {engagement_data.get(field)}")
            
            # Handle role relationships
            role_fields = ['lcsp_role_id', 'engagement_partner_role_id', 'engagement_manager_role_id']
            for field in role_fields:
                if field in engagement_data and pd.isna(engagement_data[field]):
                    engagement_data[field] = None
            
            # Validate against ontology
            success, validation_errors = validator.validate_model(Engagement, engagement_data)
            if not success:
                errors.extend([
                    f"Error in engagement {engagement_data.get('engagement_id')} for period {engagement_data.get('snapshot_period')}: {err}" 
                    for err in validation_errors
                ])
                continue
                
            # Create Pydantic model
            engagement_model = Engagement(**engagement_data)
            engagement_records.append(engagement_model.dict())
        
        # Raise error if validation failed
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Engagement facts validation failed:\n{error_msg}")
        
        # Insert records into database
        if engagement_records:
            db_manager.insert_engagement_facts(engagement_records)
            logger.info(f"Inserted {len(engagement_records)} engagement fact records")
    else:
        logger.warning("Required engagement columns missing, skipping engagement facts")

def calculate_metrics(db_manager):
    """Calculate derived metrics for all snapshot periods."""
    logger.info("Calculating derived metrics")
    
    # Get all snapshot periods
    periods = db_manager.execute_query("SELECT DISTINCT snapshot_period FROM dim_time ORDER BY snapshot_period")
    
    for period_row in periods:
        period = period_row[0]
        logger.info(f"Calculating metrics for period {period}")
        
        # Calculate metrics for this period
        db_manager.calculate_metrics_for_period(period)

if __name__ == "__main__":
    load_test_data() 