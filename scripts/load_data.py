"""
Load data into the Analytics-STeeR database

This script loads data from CSV files into the database using Pydantic models.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from models import (
    Time, Client, Person, Role, Engagement,
    PopMetrics, SeasonalityMetrics
)
from db_manager import DatabaseManager
from validation import OntologyValidator

class DataLoader:
    def __init__(self, db_path: str = "data/analytics_steer.duckdb"):
        """Initialize the data loader"""
        self.db = DatabaseManager(db_path)
        self.validator = OntologyValidator()
    
    def load_time_dimension(self, csv_path: str) -> None:
        """Load time dimension data"""
        print(f"Loading time dimension from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert date strings to datetime objects
        df['debut_periode'] = pd.to_datetime(df['debut_periode']).dt.date
        df['fin_periode'] = pd.to_datetime(df['fin_periode']).dt.date
        
        # Validate data against ontology
        errors = []
        for _, row in df.iterrows():
            row_errors = self.validator.validate_time_data(row.to_dict())
            if row_errors:
                errors.extend(row_errors)
        
        if errors:
            raise ValueError("Time dimension validation failed:\n" + "\n".join(errors))
        
        # Create Pydantic models
        time_models = [Time(**row) for _, row in df.iterrows()]
        
        # Insert into database
        self.db.insert_many(time_models)
    
    def load_client_dimension(self, csv_path: str) -> None:
        """Load client dimension data"""
        print(f"Loading client dimension from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Validate data against ontology
        errors = []
        for _, row in df.iterrows():
            row_errors = self.validator.validate_client_data(row.to_dict())
            if row_errors:
                errors.extend(row_errors)
        
        if errors:
            raise ValueError("Client dimension validation failed:\n" + "\n".join(errors))
        
        # Create Pydantic models
        client_models = [Client(**row) for _, row in df.iterrows()]
        
        # Insert into database
        self.db.insert_many(client_models)
    
    def load_person_dimension(self, csv_path: str) -> None:
        """Load person dimension data"""
        print(f"Loading person dimension from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Validate data against ontology
        errors = []
        for _, row in df.iterrows():
            row_errors = self.validator.validate_person_data(row.to_dict())
            if row_errors:
                errors.extend(row_errors)
        
        if errors:
            raise ValueError("Person dimension validation failed:\n" + "\n".join(errors))
        
        # Create Pydantic models
        person_models = [Person(**row) for _, row in df.iterrows()]
        
        # Insert into database
        self.db.insert_many(person_models)
    
    def load_role_dimension(self, csv_path: str) -> None:
        """Load role dimension data"""
        print(f"Loading role dimension from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert date strings to datetime objects
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date']).dt.date
        
        # Validate data against ontology
        errors = []
        for _, row in df.iterrows():
            row_errors = self.validator.validate_role_data(row.to_dict())
            if row_errors:
                errors.extend(row_errors)
        
        if errors:
            raise ValueError("Role dimension validation failed:\n" + "\n".join(errors))
        
        # Create Pydantic models
        role_models = [Role(**row) for _, row in df.iterrows()]
        
        # Insert into database
        self.db.insert_many(role_models)
    
    def load_engagement_facts(self, csv_path: str) -> None:
        """Load engagement fact data"""
        print(f"Loading engagement facts from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Validate data against ontology
        errors = []
        for _, row in df.iterrows():
            row_errors = self.validator.validate_engagement_data(row.to_dict())
            if row_errors:
                errors.extend(row_errors)
        
        if errors:
            raise ValueError("Engagement facts validation failed:\n" + "\n".join(errors))
        
        # Create Pydantic models
        engagement_models = [Engagement(**row) for _, row in df.iterrows()]
        
        # Insert into database
        self.db.insert_many(engagement_models)
    
    def calculate_metrics(self, snapshot_period: str) -> None:
        """Calculate derived metrics for a period"""
        print(f"Calculating metrics for period {snapshot_period}...")
        self.db.calculate_metrics(snapshot_period)
    
    def close(self) -> None:
        """Close the database connection"""
        self.db.close()

def load_all_data(data_dir: str = "data") -> None:
    """Load all data from CSV files in the data directory"""
    data_dir = Path(data_dir)
    loader = DataLoader()
    
    try:
        # Load dimensions
        loader.load_time_dimension(str(data_dir / "time_dimension.csv"))
        loader.load_client_dimension(str(data_dir / "client_dimension.csv"))
        loader.load_person_dimension(str(data_dir / "person_dimension.csv"))
        loader.load_role_dimension(str(data_dir / "role_dimension.csv"))
        
        # Load facts
        loader.load_engagement_facts(str(data_dir / "engagement_facts.csv"))
        
        # Calculate metrics for all periods
        periods = loader.db.conn.execute("""
            SELECT DISTINCT snapshot_period 
            FROM fact_engagements 
            ORDER BY snapshot_period
        """).fetchall()
        
        for period in periods:
            loader.calculate_metrics(period[0])
        
        print("Data loading completed successfully!")
        
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise
    finally:
        loader.close()

if __name__ == "__main__":
    load_all_data() 