"""
Schema Generator for Analytics-STeeR

This module generates DuckDB schema from Pydantic models.
"""

from typing import Type, Dict, Any, Optional
from pydantic import BaseModel
from datetime import date
import duckdb
from scripts.models import (
    Time, Client, Person, Role, Engagement,
    PopMetrics, SeasonalityMetrics
)

class SchemaGenerator:
    """Generates DuckDB schema from Pydantic models"""
    
    TYPE_MAPPING = {
        str: "VARCHAR",
        int: "INTEGER",
        float: "DECIMAL(15,2)",
        bool: "BOOLEAN",
        date: "DATE",
        "Optional[str]": "VARCHAR",
        "Optional[int]": "INTEGER",
        "Optional[float]": "DECIMAL(15,2)",
        "Optional[bool]": "BOOLEAN",
        "Optional[date]": "DATE"
    }
    
    def __init__(self):
        self.schema = {}
    
    def _get_sql_type(self, field_type: Type) -> str:
        """Convert Python type to SQL type"""
        type_str = str(field_type)
        if type_str in self.TYPE_MAPPING:
            return self.TYPE_MAPPING[type_str]
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
            return self.TYPE_MAPPING[f"Optional[{str(field_type.__args__[0])}]"]
        return "VARCHAR"
    
    def _generate_table_schema(self, model: Type[BaseModel], table_name: str) -> str:
        """Generate CREATE TABLE statement for a model"""
        fields = []
        composite_pk = []
        
        for field_name, field in model.model_fields.items():
            sql_type = self._get_sql_type(field.annotation)
            constraints = []
            
            # Add primary key constraint - only for the main identifier of each table
            if (field_name == "snapshot_period" and table_name == "dim_time") or \
               (field_name == "client_id" and table_name == "dim_clients") or \
               (field_name == "person_id" and table_name == "dim_people") or \
               (field_name == "role_id" and table_name == "dim_roles") or \
               (field_name == "engagement_id" and table_name == "fact_engagements"):
                constraints.append("PRIMARY KEY")
            
            # Collect composite primary key fields for fact tables
            if table_name in ["fact_pop_metrics", "fact_seasonality_metrics"]:
                if field_name in ["engagement_id", "snapshot_period"]:
                    composite_pk.append(field_name)
            
            # Add foreign key constraints
            if field_name == "client_id" and table_name != "dim_clients":
                constraints.append(f"REFERENCES dim_clients(client_id)")
            elif field_name == "person_id" and table_name != "dim_people":
                constraints.append(f"REFERENCES dim_people(person_id)")
            elif field_name == "role_id" and table_name != "dim_roles":
                constraints.append(f"REFERENCES dim_roles(role_id)")
            elif field_name == "snapshot_period" and table_name != "dim_time":
                constraints.append(f"REFERENCES dim_time(snapshot_period)")
            
            # Add NOT NULL constraint for required fields
            if field.is_required:
                constraints.append("NOT NULL")
            
            field_def = f"{field_name} {sql_type}"
            if constraints:
                field_def += " " + " ".join(constraints)
            
            fields.append(field_def)
        
        # Add composite primary key constraint if needed
        if composite_pk:
            fields.append(f"PRIMARY KEY ({', '.join(composite_pk)})")
        
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {',\n            '.join(fields)}
        )
        """
    
    def generate_schema(self) -> Dict[str, str]:
        """Generate complete database schema"""
        return {
            "dim_time": self._generate_table_schema(Time, "dim_time"),
            "dim_clients": self._generate_table_schema(Client, "dim_clients"),
            "dim_people": self._generate_table_schema(Person, "dim_people"),
            "dim_roles": self._generate_table_schema(Role, "dim_roles"),
            "fact_engagements": self._generate_table_schema(Engagement, "fact_engagements"),
            "fact_pop_metrics": self._generate_table_schema(PopMetrics, "fact_pop_metrics"),
            "fact_seasonality_metrics": self._generate_table_schema(SeasonalityMetrics, "fact_seasonality_metrics")
        }
    
    def create_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create the schema in the database"""
        schema = self.generate_schema()
        
        # Create tables in order to satisfy foreign key constraints
        table_order = [
            "dim_time",      # No foreign keys
            "dim_clients",   # No foreign keys
            "dim_people",    # No foreign keys
            "dim_roles",     # References dim_people, dim_clients
            "fact_engagements", # References dim_clients, dim_time
            "fact_pop_metrics", # References fact_engagements, dim_time
            "fact_seasonality_metrics" # References fact_engagements, dim_time
        ]
        
        for table_name in table_order:
            create_stmt = schema[table_name]
            conn.execute(create_stmt)
            print(f"Created table: {table_name}")

if __name__ == "__main__":
    # Test the schema generator
    conn = duckdb.connect(":memory:")
    generator = SchemaGenerator()
    generator.create_schema(conn)
    
    # Print table information
    tables = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
    for table in tables:
        print(f"\nTable: {table[0]}")
        columns = conn.execute(f"PRAGMA table_info({table[0]})").fetchall()
        for col in columns:
            print(f"  {col[1]}: {col[2]}") 