"""
Database manager for Analytics-STeeR

This module provides database operations using Pydantic models and DuckDB.
Following DuckDB best practices:
1. Use VARCHAR with explicit lengths for string columns
2. Use DECIMAL for financial data with appropriate precision
3. Use CHECK constraints for data validation
4. Use FOREIGN KEY constraints for referential integrity
5. Use PRIMARY KEY constraints for unique identification
6. Use appropriate data types for temporal data
7. Use appropriate indexes for frequently queried columns
8. Use PRAGMA statements for optimization where available
9. Use prepared statements for better performance
10. Use transactions for data integrity
"""

import duckdb
from pathlib import Path
import os
import json
from typing import List, Type, TypeVar, Generic, Optional, Dict, Any
from datetime import date
import pandas as pd
from scripts.models import (
    Time, Client, Person, Role, Engagement,
    PopMetrics, SeasonalityMetrics
)

T = TypeVar('T')

# Current database schema version
DB_VERSION = "1.0.0"

class DatabaseManager:
    def __init__(self, db_path: str = "data/analytics_steer.duckdb", initialize: bool = True):
        """Initialize the database manager with DuckDB-specific settings
        
        Args:
            db_path: Path to the DuckDB database file
            initialize: Whether to initialize the database schema if it doesn't exist
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database
        self.conn = duckdb.connect(db_path)
        
        # Configure DuckDB settings
        self._configure_duckdb()
        
        # Initialize schema if requested
        if initialize:
            db_exists = self._check_if_db_exists()
            if not db_exists:
                self._initialize_database()
            else:
                self._check_and_update_schema()
    
    def _configure_duckdb(self) -> None:
        """Configure DuckDB-specific settings"""
        self.conn.execute("""
            -- Enable parallel processing if available
            SET threads TO 4;
            
            -- Set memory limit if supported
            PRAGMA memory_limit='4GB';
            
            -- Set temp directory if needed
            PRAGMA temp_directory='data/temp';
        """)
    
    def _check_if_db_exists(self) -> bool:
        """Check if the database has been initialized"""
        try:
            # Check if the metadata table exists
            result = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='db_metadata'
            """).fetchall()
            return len(result) > 0
        except:
            return False
    
    def _initialize_database(self) -> None:
        """Initialize the database with the schema and metadata"""
        # Start a transaction for schema creation
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            # Create metadata table first
            self.conn.execute("""
                CREATE TABLE db_metadata (
                    key VARCHAR(100) PRIMARY KEY,
                    value VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert version information
            self.conn.execute("""
                INSERT INTO db_metadata (key, value) 
                VALUES ('version', ?), ('created_at', CURRENT_TIMESTAMP)
            """, [DB_VERSION])
            
            # Create all tables
            self._create_tables()
            
            # Commit the transaction
            self.conn.execute("COMMIT")
            print(f"Database initialized with version {DB_VERSION}")
            
        except Exception as e:
            # Rollback on error
            self.conn.execute("ROLLBACK")
            raise Exception(f"Database initialization failed: {str(e)}")
    
    def _check_and_update_schema(self) -> None:
        """Check if schema needs updating and update if necessary"""
        try:
            # Get current version
            current_version = self.conn.execute("""
                SELECT value FROM db_metadata WHERE key = 'version'
            """).fetchone()[0]
            
            # Compare versions and update if needed
            if current_version != DB_VERSION:
                # For now, just update the version
                # In a real system, you would have migration scripts
                self.conn.execute("""
                    UPDATE db_metadata 
                    SET value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE key = 'version'
                """, [DB_VERSION])
                print(f"Updated database from version {current_version} to {DB_VERSION}")
        except Exception as e:
            print(f"Error checking database version: {str(e)}")
    
    def _create_tables(self) -> None:
        """Create database tables in proper order to satisfy foreign key constraints"""
        # Create tables in dependency order
        
        # 1. First, create tables with no dependencies
        self._create_time_dimension()
        self._create_client_dimension()
        self._create_person_dimension()
        
        # 2. Then create tables dependent on the first batch
        self._create_role_dimension()
        self._create_engagement_facts()
        
        # 3. Finally create tables dependent on the second batch
        self._create_pop_metrics()
        self._create_seasonality_metrics()
    
    def _create_time_dimension(self) -> None:
        """Create time dimension table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_time (
                snapshot_period VARCHAR(7) PRIMARY KEY,
                year INTEGER NOT NULL CHECK (year >= 2000 AND year <= 2100),
                period INTEGER NOT NULL CHECK (period >= 1 AND period <= 12),
                period_type VARCHAR(10) NOT NULL DEFAULT 'month' CHECK (period_type IN ('month', 'quarter', 'semester')),
                fiscal_year INTEGER NOT NULL CHECK (fiscal_year >= 2000 AND fiscal_year <= 2100),
                fiscal_period INTEGER NOT NULL CHECK (fiscal_period >= 1 AND fiscal_period <= 12),
                debut_periode DATE NOT NULL,
                fin_periode DATE NOT NULL,
                is_current_period BOOLEAN NOT NULL DEFAULT FALSE,
                fiscal_year_start_month INTEGER NOT NULL DEFAULT 1 CHECK (fiscal_year_start_month >= 1 AND fiscal_year_start_month <= 12),
                CONSTRAINT valid_fiscal_period CHECK (
                    (period_type = 'month' AND fiscal_period <= 12) OR
                    (period_type = 'quarter' AND fiscal_period <= 4) OR
                    (period_type = 'semester' AND fiscal_period <= 2)
                )
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time_year_period 
            ON dim_time(year, period);
        """)
    
    def _create_client_dimension(self) -> None:
        """Create client dimension table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_clients (
                client_id VARCHAR(50) PRIMARY KEY,
                client_name VARCHAR(255) NOT NULL,
                client_group_name VARCHAR(50),
                direct_parent_id VARCHAR(50),
                industry_sector VARCHAR(100),
                industry VARCHAR(100),
                sector VARCHAR(100),
                region VARCHAR(100),
                pole VARCHAR(100),
                FOREIGN KEY (direct_parent_id) REFERENCES dim_clients(client_id)
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_clients_name 
            ON dim_clients(client_name);
            
            CREATE INDEX IF NOT EXISTS idx_clients_group 
            ON dim_clients(client_group_name);
        """)
    
    def _create_person_dimension(self) -> None:
        """Create person dimension table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_people (
                person_id VARCHAR(50) PRIMARY KEY,
                matricule VARCHAR(50) UNIQUE NOT NULL,
                full_name VARCHAR(255) NOT NULL
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_people_name 
            ON dim_people(full_name);
            
            CREATE INDEX IF NOT EXISTS idx_people_matricule 
            ON dim_people(matricule);
        """)
    
    def _create_role_dimension(self) -> None:
        """Create role dimension table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_roles (
                role_id VARCHAR(50) PRIMARY KEY,
                role_type VARCHAR(50) NOT NULL CHECK (role_type IN ('LCSP', 'EngagementPartner', 'EngagementManager', 
                                                                    'ParticipatingPartner', 'ParticipatingManager',
                                                                    'BillingPartner', 'BillingManager', 
                                                                    'EngagementController', 'Consultant')),
                authority_level INTEGER NOT NULL CHECK (authority_level >= 1 AND authority_level <= 8),
                start_date DATE NOT NULL,
                end_date DATE,
                person_id VARCHAR(50) NOT NULL,
                client_id VARCHAR(50),
                engagement_id VARCHAR(50),
                under_authority_of VARCHAR(50),
                FOREIGN KEY (person_id) REFERENCES dim_people(person_id),
                FOREIGN KEY (client_id) REFERENCES dim_clients(client_id),
                FOREIGN KEY (under_authority_of) REFERENCES dim_roles(role_id),
                CONSTRAINT valid_role_authority CHECK (
                    (role_type = 'LCSP' AND authority_level = 1) OR
                    (role_type = 'EngagementPartner' AND authority_level = 2) OR
                    (role_type = 'EngagementManager' AND authority_level = 3) OR
                    (role_type = 'ParticipatingPartner' AND authority_level = 4) OR
                    (role_type = 'ParticipatingManager' AND authority_level = 5) OR
                    (role_type = 'BillingPartner' AND authority_level = 6) OR
                    (role_type = 'BillingManager' AND authority_level = 7) OR
                    (role_type = 'EngagementController' AND authority_level = 8)
                ),
                CONSTRAINT valid_dates CHECK (
                    end_date IS NULL OR end_date >= start_date
                )
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_roles_person 
            ON dim_roles(person_id);
            
            CREATE INDEX IF NOT EXISTS idx_roles_client 
            ON dim_roles(client_id);
            
            CREATE INDEX IF NOT EXISTS idx_roles_type 
            ON dim_roles(role_type);
        """)
    
    def _create_engagement_facts(self) -> None:
        """Create engagement facts table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_engagements (
                engagement_id VARCHAR(50),
                snapshot_period VARCHAR(7),
                client_id VARCHAR(50),
                master_engagement VARCHAR(50),
                net_revenue DECIMAL(15,2) NOT NULL CHECK (net_revenue >= 0),
                total_costs DECIMAL(15,2) NOT NULL CHECK (total_costs >= 0),
                net_margin DECIMAL(15,2) NOT NULL,
                margin_percentage DECIMAL(5,2) NOT NULL CHECK (margin_percentage >= 0 AND margin_percentage <= 100),
                PRIMARY KEY (engagement_id, snapshot_period),
                FOREIGN KEY (client_id) REFERENCES dim_clients(client_id),
                FOREIGN KEY (snapshot_period) REFERENCES dim_time(snapshot_period),
                CONSTRAINT valid_margin CHECK (
                    net_margin = net_revenue - total_costs AND
                    (net_revenue = 0 OR margin_percentage = (net_margin / net_revenue) * 100)
                )
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_engagements_client 
            ON fact_engagements(client_id);
            
            CREATE INDEX IF NOT EXISTS idx_engagements_period 
            ON fact_engagements(snapshot_period);
            
            CREATE INDEX IF NOT EXISTS idx_engagements_master 
            ON fact_engagements(master_engagement);
        """)
    
    def _create_pop_metrics(self) -> None:
        """Create PoP metrics table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_pop_metrics (
                engagement_id VARCHAR(50),
                snapshot_period VARCHAR(7),
                previous_period_revenue DECIMAL(15,2),
                previous_year_revenue DECIMAL(15,2),
                pop_revenue_growth DECIMAL(5,2),
                yoy_revenue_growth DECIMAL(5,2),
                pop_margin_growth DECIMAL(5,2),
                PRIMARY KEY (engagement_id, snapshot_period),
                FOREIGN KEY (engagement_id, snapshot_period) 
                    REFERENCES fact_engagements(engagement_id, snapshot_period)
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pop_metrics_engagement 
            ON fact_pop_metrics(engagement_id, snapshot_period);
        """)
    
    def _create_seasonality_metrics(self) -> None:
        """Create seasonality metrics table"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_seasonality_metrics (
                engagement_id VARCHAR(50),
                snapshot_period VARCHAR(7),
                seasonal_factor DECIMAL(10,4),
                seasonally_adjusted_revenue DECIMAL(15,2),
                seasonally_adjusted_growth DECIMAL(5,2),
                PRIMARY KEY (engagement_id, snapshot_period),
                FOREIGN KEY (engagement_id, snapshot_period) 
                    REFERENCES fact_engagements(engagement_id, snapshot_period)
            )
        """)
        
        # Create index on frequently queried columns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seasonality_metrics_engagement 
            ON fact_seasonality_metrics(engagement_id, snapshot_period);
        """)
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the database schema"""
        tables = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
        
        schema_info = {
            "version": self.conn.execute("""
                SELECT value FROM db_metadata WHERE key = 'version'
            """).fetchone()[0],
            "tables": {}
        }
        
        for table in tables:
            table_name = table[0]
            if table_name != 'db_metadata':
                columns = self.conn.execute(f"""
                    PRAGMA table_info({table_name})
                """).fetchall()
                
                schema_info["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "notnull": col[3],
                            "pk": col[5]
                        } for col in columns
                    ]
                }
        
        return schema_info
    
    def verify_db_pydantic_sync(self) -> List[str]:
        """Verify that the database schema is in sync with Pydantic models"""
        discrepancies = []
        
        # Map model fields to table columns
        model_mappings = {
            'Time': ('dim_time', Time),
            'Client': ('dim_clients', Client),
            'Person': ('dim_people', Person),
            'Role': ('dim_roles', Role),
            'Engagement': ('fact_engagements', Engagement),
            'PopMetrics': ('fact_pop_metrics', PopMetrics),
            'SeasonalityMetrics': ('fact_seasonality_metrics', SeasonalityMetrics)
        }
        
        for model_name, (table_name, model_class) in model_mappings.items():
            # Get model fields
            model_fields = model_class.__annotations__
            
            # Get table columns
            columns = self.conn.execute(f"""
                PRAGMA table_info({table_name})
            """).fetchall()
            db_columns = {col[1]: col[2] for col in columns}
            
            # Compare fields to columns
            for field_name in model_fields:
                # Skip internal fields
                if field_name.startswith('_'):
                    continue
                    
                # Check if field exists in database
                if field_name not in db_columns:
                    discrepancies.append(f"Field '{field_name}' in model '{model_name}' doesn't exist in table '{table_name}'")
            
            # Check for columns not in model
            for col_name in db_columns:
                field_exists = col_name in model_fields
                if not field_exists and col_name not in ['id']:  # Exclude common auto-generated fields
                    discrepancies.append(f"Column '{col_name}' in table '{table_name}' doesn't exist in model '{model_name}'")
        
        return discrepancies
    
    def insert(self, model: T) -> None:
        """Insert a Pydantic model into the database using DuckDB's efficient insert"""
        table_name = self._get_table_name(model)
        df = pd.DataFrame([model.dict()])
        self.conn.register('temp_df', df)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        self.conn.unregister('temp_df')
    
    def insert_many(self, models: List[T]) -> None:
        """Insert multiple Pydantic models into the database using DuckDB's efficient batch insert"""
        if not models:
            return
        table_name = self._get_table_name(models[0])
        df = pd.DataFrame([model.dict() for model in models])
        self.conn.register('temp_df', df)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        self.conn.unregister('temp_df')
    
    def _get_table_name(self, model: T) -> str:
        """Get the table name for a Pydantic model"""
        model_name = model.__class__.__name__
        if model_name == 'Time':
            return 'dim_time'
        elif model_name == 'Client':
            return 'dim_clients'
        elif model_name == 'Person':
            return 'dim_people'
        elif model_name == 'Role':
            return 'dim_roles'
        elif model_name == 'Engagement':
            return 'fact_engagements'
        elif model_name == 'PopMetrics':
            return 'fact_pop_metrics'
        elif model_name == 'SeasonalityMetrics':
            return 'fact_seasonality_metrics'
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def calculate_metrics(self, snapshot_period: str) -> None:
        """Calculate derived metrics for a period using DuckDB's window functions"""
        # Use a transaction for data integrity
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            # Calculate YTD metrics
            self.conn.execute("""
                WITH ytd_data AS (
                    SELECT 
                        e.engagement_id,
                        e.snapshot_period,
                        SUM(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id, t.fiscal_year 
                            ORDER BY t.fiscal_period 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) as ytd_net_revenue,
                        SUM(e.net_margin) OVER (
                            PARTITION BY e.engagement_id, t.fiscal_year 
                            ORDER BY t.fiscal_period 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) as ytd_net_margin
                    FROM fact_engagements e
                    JOIN dim_time t ON e.snapshot_period = t.snapshot_period
                    WHERE t.fiscal_period <= (
                        SELECT fiscal_period 
                        FROM dim_time 
                        WHERE snapshot_period = ?
                    )
                    AND t.fiscal_year = (
                        SELECT fiscal_year 
                        FROM dim_time 
                        WHERE snapshot_period = ?
                    )
                )
                UPDATE fact_engagements e
                SET 
                    ytd_net_revenue = y.ytd_net_revenue,
                    ytd_net_margin = y.ytd_net_margin,
                    ytd_margin_percentage = CASE 
                        WHEN y.ytd_net_revenue > 0 
                        THEN (y.ytd_net_margin / y.ytd_net_revenue) * 100 
                        ELSE 0 
                    END
                FROM ytd_data y
                WHERE e.engagement_id = y.engagement_id 
                AND e.snapshot_period = y.snapshot_period
            """, (snapshot_period, snapshot_period))
            
            # Calculate PoP metrics - first ensure we delete any existing records
            self.conn.execute("""
                DELETE FROM fact_pop_metrics
                WHERE snapshot_period = ?
            """, (snapshot_period,))
            
            # Now insert new pop metric records
            self.conn.execute("""
                WITH pop_data AS (
                    SELECT 
                        e.engagement_id,
                        e.snapshot_period,
                        LAG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        ) as previous_period_revenue,
                        LAG(e.net_revenue, 12) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        ) as previous_year_revenue,
                        LAG(e.net_margin) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        ) as previous_period_margin
                    FROM fact_engagements e
                )
                INSERT INTO fact_pop_metrics (
                    engagement_id, snapshot_period, 
                    previous_period_revenue, previous_year_revenue,
                    pop_revenue_growth, yoy_revenue_growth, pop_margin_growth
                )
                SELECT 
                    p.engagement_id,
                    p.snapshot_period,
                    p.previous_period_revenue,
                    p.previous_year_revenue,
                    CASE 
                        WHEN p.previous_period_revenue > 0 
                        THEN ((e.net_revenue - p.previous_period_revenue) / p.previous_period_revenue) * 100 
                        ELSE NULL 
                    END as pop_revenue_growth,
                    CASE 
                        WHEN p.previous_year_revenue > 0 
                        THEN ((e.net_revenue - p.previous_year_revenue) / p.previous_year_revenue) * 100 
                        ELSE NULL 
                    END as yoy_revenue_growth,
                    CASE 
                        WHEN p.previous_period_margin > 0 
                        THEN ((e.net_margin - p.previous_period_margin) / p.previous_period_margin) * 100 
                        ELSE NULL 
                    END as pop_margin_growth
                FROM pop_data p
                JOIN fact_engagements e ON p.engagement_id = e.engagement_id 
                    AND p.snapshot_period = e.snapshot_period
                WHERE p.snapshot_period = ?
            """, (snapshot_period,))
            
            # Clean up any existing seasonality metrics
            self.conn.execute("""
                DELETE FROM fact_seasonality_metrics
                WHERE snapshot_period = ?
            """, (snapshot_period,))
            
            # Insert new seasonality metrics
            self.conn.execute("""
                WITH seasonal_data AS (
                    SELECT 
                        e.engagement_id,
                        e.snapshot_period,
                        t.fiscal_period,
                        AVG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id, t.fiscal_period 
                            ORDER BY t.fiscal_year 
                            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                        ) as avg_historical_revenue,
                        AVG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY t.fiscal_year 
                            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                        ) as avg_total_revenue
                    FROM fact_engagements e
                    JOIN dim_time t ON e.snapshot_period = t.snapshot_period
                )
                INSERT INTO fact_seasonality_metrics (
                    engagement_id, snapshot_period,
                    seasonal_factor, seasonally_adjusted_revenue,
                    seasonally_adjusted_growth
                )
                SELECT 
                    s.engagement_id,
                    s.snapshot_period,
                    CASE 
                        WHEN s.avg_total_revenue > 0 
                        THEN s.avg_historical_revenue / s.avg_total_revenue 
                        ELSE NULL 
                    END as seasonal_factor,
                    CASE 
                        WHEN s.avg_historical_revenue > 0 
                        THEN e.net_revenue / (s.avg_historical_revenue / s.avg_total_revenue) 
                        ELSE NULL 
                    END as seasonally_adjusted_revenue,
                    CASE 
                        WHEN LAG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        ) > 0 
                        THEN ((e.net_revenue - LAG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        )) / LAG(e.net_revenue) OVER (
                            PARTITION BY e.engagement_id 
                            ORDER BY e.snapshot_period
                        )) * 100 
                        ELSE NULL 
                    END as seasonally_adjusted_growth
                FROM seasonal_data s
                JOIN fact_engagements e ON s.engagement_id = e.engagement_id 
                    AND s.snapshot_period = e.snapshot_period
                WHERE s.snapshot_period = ?
            """, (snapshot_period,))
            
            # Commit the transaction
            self.conn.execute("COMMIT")
            
        except Exception as e:
            # Rollback the transaction on error
            self.conn.execute("ROLLBACK")
            raise e
    
    def close(self) -> None:
        """Close the database connection"""
        self.conn.close() 