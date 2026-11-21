"""
Schema Manager for Analytics-STeeR

This module implements the ontology-defined schema and provides functions for:
- Schema creation and validation
- Period calculations
- Financial metrics calculations
- Data type management
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import duckdb
from pathlib import Path
from .schema_generator import SchemaGenerator

class SchemaManager:
    def __init__(self, db_path: str = "data/analytics_steer.duckdb"):
        """Initialize the schema manager with database connection"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.ontology = self._load_ontology()
        self.schema_generator = SchemaGenerator()
        
    def _load_ontology(self) -> Dict:
        """Load the ontology definition"""
        ontology_path = Path(__file__).parent.parent / "ontology" / "analytics_steer_ontology.jsonld"
        with open(ontology_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_schema(self) -> None:
        """Create the database schema based on ontology and Pydantic models"""
        self.schema_generator.create_schema(self.conn)
    
    def calculate_period_metrics(self, snapshot_period: str) -> None:
        """Calculate period metrics including YTD and PoP"""
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
        
        # Calculate PoP metrics
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
                pop_revenue_growth, yoy_revenue_growth,
                pop_margin_growth
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
            ON CONFLICT (engagement_id, snapshot_period) DO UPDATE SET
                previous_period_revenue = EXCLUDED.previous_period_revenue,
                previous_year_revenue = EXCLUDED.previous_year_revenue,
                pop_revenue_growth = EXCLUDED.pop_revenue_growth,
                yoy_revenue_growth = EXCLUDED.yoy_revenue_growth,
                pop_margin_growth = EXCLUDED.pop_margin_growth
        """)
    
    def validate_schema(self) -> List[str]:
        """Validate the schema against the ontology"""
        validation_errors = []
        
        # Check required tables
        required_tables = [
            'dim_time', 'dim_clients', 'dim_people', 
            'dim_roles', 'fact_engagements',
            'fact_pop_metrics', 'fact_seasonality_metrics'
        ]
        
        for table in required_tables:
            result = self.conn.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{table}'
            """).fetchone()
            if result[0] == 0:
                validation_errors.append(f"Missing required table: {table}")
        
        return validation_errors
    
    def close(self) -> None:
        """Close the database connection"""
        self.conn.close()

if __name__ == "__main__":
    # Example usage
    schema_manager = SchemaManager()
    schema_manager.create_schema()
    
    # Validate schema
    errors = schema_manager.validate_schema()
    if errors:
        print("Schema validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Schema validation successful")
    
    schema_manager.close() 