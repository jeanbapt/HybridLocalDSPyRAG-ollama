"""
Main execution script for Analytics-STeeR

This script orchestrates the entire data loading and analytics process.
"""

import os
import argparse
import logging
from pathlib import Path
from load_test_data import load_test_data
from db_manager import DatabaseManager
from test_data_loading import run_tests
from validation import OntologyValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analytics_steer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analytics_steer")

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ["data", "reports", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info(f"Created directories: {', '.join(directories)}")

def validate_environment():
    """Validate the environment and required files"""
    required_files = [
        "test-structure.csv",
        "ontology/analytics_steer_ontology.jsonld"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    # Validate ontology file
    try:
        validator = OntologyValidator()
        logger.info("Ontology file loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ontology: {str(e)}")
        return False
    
    return True

def run_data_pipeline(csv_path, db_path, run_validation=True, clean=False):
    """Run the full data pipeline"""
    # Clean existing database if requested
    if clean and os.path.exists(db_path):
        logger.info(f"Removing existing database: {db_path}")
        os.remove(db_path)
    
    # Run unit tests if validation is enabled
    if run_validation:
        logger.info("Running validation tests...")
        run_tests()
    
    # Load data
    logger.info(f"Loading data from {csv_path} into {db_path}...")
    try:
        load_test_data(csv_path, db_path)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False
    
    # Run database validation
    try:
        db = DatabaseManager(db_path)
        
        # Query and log some basic statistics
        stats = {
            "time_periods": db.conn.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0],
            "clients": db.conn.execute("SELECT COUNT(*) FROM dim_clients").fetchone()[0],
            "people": db.conn.execute("SELECT COUNT(*) FROM dim_people").fetchone()[0],
            "roles": db.conn.execute("SELECT COUNT(*) FROM dim_roles").fetchone()[0],
            "engagements": db.conn.execute("SELECT COUNT(*) FROM fact_engagements").fetchone()[0]
        }
        
        logger.info("Database statistics:")
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
        
        db.close()
    except Exception as e:
        logger.error(f"Error validating database: {str(e)}")
        return False
    
    return True

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Analytics-STeeR Data Pipeline")
    
    parser.add_argument(
        "--csv-path", 
        default="test-structure.csv",
        help="Path to the CSV file containing the data"
    )
    
    parser.add_argument(
        "--db-path", 
        default="data/analytics_steer.duckdb",
        help="Path to the DuckDB database file"
    )
    
    parser.add_argument(
        "--no-validation", 
        action="store_true",
        help="Skip validation tests"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean existing database before loading data"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    logger.info("=== Starting Analytics-STeeR Pipeline ===")
    
    # Setup directories
    setup_directories()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed, exiting")
        return 1
    
    # Run the data pipeline
    success = run_data_pipeline(
        args.csv_path,
        args.db_path,
        not args.no_validation,
        args.clean
    )
    
    if success:
        logger.info("=== Pipeline completed successfully ===")
        return 0
    else:
        logger.error("=== Pipeline failed ===")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 