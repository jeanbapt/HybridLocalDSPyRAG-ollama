"""
Test script for schema generator
"""

import unittest
import duckdb
from scripts.schema_generator import SchemaGenerator
from scripts.models import (
    Time, Client, Person, Role, Engagement,
    PopMetrics, SeasonalityMetrics
)

class TestSchemaGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.conn = duckdb.connect(":memory:")
        self.generator = SchemaGenerator()
    
    def test_schema_generation(self):
        """Test that schema is generated correctly"""
        # Generate and create schema
        self.generator.create_schema(self.conn)
        
        # Get all tables
        tables = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        # Check all required tables exist
        required_tables = {
            'dim_time', 'dim_clients', 'dim_people', 
            'dim_roles', 'fact_engagements',
            'fact_pop_metrics', 'fact_seasonality_metrics'
        }
        created_tables = {table[0] for table in tables}
        self.assertEqual(created_tables, required_tables)
        
        # Check each table's structure
        self._check_time_table()
        self._check_clients_table()
        self._check_people_table()
        self._check_roles_table()
        self._check_engagements_table()
        self._check_pop_metrics_table()
        self._check_seasonality_metrics_table()
    
    def _check_time_table(self):
        """Check dim_time table structure"""
        columns = self.conn.execute("PRAGMA table_info(dim_time)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'snapshot_period', 'year', 'period', 'period_type',
            'fiscal_year', 'fiscal_period', 'debut_periode',
            'fin_periode', 'is_current_period', 'fiscal_year_start_month'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check primary key
        pk = self.conn.execute("""
            SELECT name FROM pragma_table_info('dim_time')
            WHERE pk = 1
        """).fetchone()
        self.assertEqual(pk[0], 'snapshot_period')
    
    def _check_clients_table(self):
        """Check dim_clients table structure"""
        columns = self.conn.execute("PRAGMA table_info(dim_clients)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'client_id', 'client_name', 'client_group_name',
            'direct_parent_id', 'industry_sector', 'industry',
            'sector', 'region', 'pole'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check primary key
        pk = self.conn.execute("""
            SELECT name FROM pragma_table_info('dim_clients')
            WHERE pk = 1
        """).fetchone()
        self.assertEqual(pk[0], 'client_id')
    
    def _check_people_table(self):
        """Check dim_people table structure"""
        columns = self.conn.execute("PRAGMA table_info(dim_people)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {'person_id', 'matricule', 'full_name'}
        self.assertEqual(column_names, expected_columns)
        
        # Check primary key
        pk = self.conn.execute("""
            SELECT name FROM pragma_table_info('dim_people')
            WHERE pk = 1
        """).fetchone()
        self.assertEqual(pk[0], 'person_id')
    
    def _check_roles_table(self):
        """Check dim_roles table structure"""
        columns = self.conn.execute("PRAGMA table_info(dim_roles)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'role_id', 'role_type', 'authority_level',
            'start_date', 'end_date', 'person_id',
            'client_id', 'engagement_id', 'under_authority_of'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check primary key
        pk = self.conn.execute("""
            SELECT name FROM pragma_table_info('dim_roles')
            WHERE pk = 1
        """).fetchone()
        self.assertEqual(pk[0], 'role_id')
    
    def _check_engagements_table(self):
        """Check fact_engagements table structure"""
        columns = self.conn.execute("PRAGMA table_info(fact_engagements)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'engagement_id', 'snapshot_period', 'client_id',
            'master_engagement', 'net_revenue', 'total_costs',
            'net_margin', 'margin_percentage'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check primary key
        pk = self.conn.execute("""
            SELECT name FROM pragma_table_info('fact_engagements')
            WHERE pk = 1
        """).fetchone()
        self.assertEqual(pk[0], 'engagement_id')
    
    def _check_pop_metrics_table(self):
        """Check fact_pop_metrics table structure"""
        columns = self.conn.execute("PRAGMA table_info(fact_pop_metrics)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'engagement_id', 'snapshot_period',
            'previous_period_revenue', 'previous_year_revenue',
            'pop_revenue_growth', 'yoy_revenue_growth',
            'pop_margin_growth'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check composite primary key
        pks = self.conn.execute("""
            SELECT name FROM pragma_table_info('fact_pop_metrics')
            WHERE pk > 0
            ORDER BY pk
        """).fetchall()
        self.assertEqual(len(pks), 2)
        self.assertEqual(pks[0][0], 'engagement_id')
        self.assertEqual(pks[1][0], 'snapshot_period')
    
    def _check_seasonality_metrics_table(self):
        """Check fact_seasonality_metrics table structure"""
        columns = self.conn.execute("PRAGMA table_info(fact_seasonality_metrics)").fetchall()
        column_names = {col[1] for col in columns}
        
        expected_columns = {
            'engagement_id', 'snapshot_period',
            'seasonal_factor', 'seasonally_adjusted_revenue',
            'seasonally_adjusted_growth'
        }
        self.assertEqual(column_names, expected_columns)
        
        # Check composite primary key
        pks = self.conn.execute("""
            SELECT name FROM pragma_table_info('fact_seasonality_metrics')
            WHERE pk > 0
            ORDER BY pk
        """).fetchall()
        self.assertEqual(len(pks), 2)
        self.assertEqual(pks[0][0], 'engagement_id')
        self.assertEqual(pks[1][0], 'snapshot_period')
    
    def tearDown(self):
        """Clean up test environment"""
        self.conn.close()

if __name__ == '__main__':
    unittest.main() 