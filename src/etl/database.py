"""
Database Module - PostgreSQL Connection and UPSERT Operations
Handles all database interactions for the ETL pipeline
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from config import DB_CONFIG, LOG_FILE

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL connections and operations"""
    
    def __init__(self, pool_size: int = 5):
        """Initialize database connection pool"""
        try:
            self.pool = SimpleConnectionPool(
                1,
                pool_size,
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            logger.info(f"Database connection pool initialized with {pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction error: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def execute_query(self, query: str, params: Tuple = None, fetch: bool = False):
        """Execute a database query"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cursor.execute(query, params)
                if fetch:
                    result = cursor.fetchall()
                    return result
                else:
                    return cursor.rowcount
            except Exception as e:
                logger.error(f"Query execution error: {e}\nQuery: {query}")
                raise
            finally:
                cursor.close()
    
    def close(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")
    
    # ========== RAW DATA OPERATIONS ==========
    
    def load_raw_data(self, data: List[Dict]) -> int:
        """Load raw CSV data into raw_data table"""
        if not data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.raw_data 
                    (kit_code, msisdn, dealer_id, offre, debit, city, governorate, 
                     postal_code, latitude, longitude, locality_name, delegation_name, creation_date)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """
                
                values = [
                    (
                        d.get('kit_code'),
                        d.get('msisdn'),
                        d.get('dealer_id'),
                        d.get('offre'),
                        d.get('debit'),
                        d.get('city'),
                        d.get('governorate'),
                        d.get('postal_code'),
                        d.get('latitude'),
                        d.get('longitude'),
                        d.get('locality_name'),
                        d.get('delegation_name'),
                        d.get('creation_date')
                    )
                    for d in data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Loaded {count} raw records into database")
                return count
            except Exception as e:
                logger.error(f"Error loading raw data: {e}")
                raise
            finally:
                cursor.close()
    
    # ========== CLEANED DATA OPERATIONS ==========
    
    def load_clean_data(self, data: List[Dict]) -> Tuple[int, int]:
        """Load cleaned data into clean_data table"""
        if not data:
            return 0, 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Separate valid and invalid records
                valid_records = [d for d in data if d.get('is_valid', True)]
                invalid_records = [d for d in data if not d.get('is_valid', True)]
                
                # Insert valid records
                if valid_records:
                    query = """
                        INSERT INTO etl_fibre.clean_data 
                        (kit_code, msisdn, dealer_id, offre, debit, city, governorate,
                         postal_code, latitude, longitude, locality_name, delegation_name,
                         creation_date, is_valid, validation_errors)
                        VALUES %s
                        ON CONFLICT (msisdn) DO UPDATE SET
                            kit_code = EXCLUDED.kit_code,
                            dealer_id = EXCLUDED.dealer_id,
                            offre = EXCLUDED.offre,
                            creation_date = EXCLUDED.creation_date
                    """
                    
                    values = [
                        (
                            d.get('kit_code'),
                            d.get('msisdn'),
                            d.get('dealer_id'),
                            d.get('offre'),
                            d.get('debit'),
                            d.get('city'),
                            d.get('governorate'),
                            d.get('postal_code'),
                            d.get('latitude'),
                            d.get('longitude'),
                            d.get('locality_name'),
                            d.get('delegation_name'),
                            d.get('creation_date'),
                            True,
                            None
                        )
                        for d in valid_records
                    ]
                    
                    execute_values(cursor, query, values, page_size=1000)
                
                # Insert invalid records for audit
                if invalid_records:
                    invalid_query = """
                        INSERT INTO etl_fibre.clean_data
                        (msisdn, kit_code, is_valid, validation_errors)
                        VALUES %s
                        ON CONFLICT (msisdn) DO NOTHING
                    """
                    
                    invalid_values = [
                        (d.get('msisdn'), d.get('kit_code'), False, d.get('validation_errors'))
                        for d in invalid_records
                    ]
                    
                    execute_values(cursor, invalid_query, invalid_values, page_size=1000)
                
                logger.info(f"Loaded {len(valid_records)} valid and {len(invalid_records)} invalid records")
                return len(valid_records), len(invalid_records)
            except Exception as e:
                logger.error(f"Error loading clean data: {e}")
                raise
            finally:
                cursor.close()
    
    # ========== DIMENSION OPERATIONS ==========
    
    def upsert_dim_temps(self, dates_data: List[Dict]) -> int:
        """UPSERT records into dim_temps"""
        if not dates_data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.dim_temps
                    (full_date, day_of_week, day_name, week_of_year, month, month_name,
                     quarter, year, is_weekend, is_holiday)
                    VALUES %s
                    ON CONFLICT (full_date) DO NOTHING
                """
                
                values = [
                    (
                        d['full_date'],
                        d['day_of_week'],
                        d['day_name'],
                        d['week_of_year'],
                        d['month'],
                        d['month_name'],
                        d['quarter'],
                        d['year'],
                        d['is_weekend'],
                        d.get('is_holiday', False)
                    )
                    for d in dates_data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Upserted {count} time dimension records")
                return count
            except Exception as e:
                logger.error(f"Error upserting dim_temps: {e}")
                raise
            finally:
                cursor.close()
    
    def upsert_dim_offres(self, offers_data: List[Dict]) -> int:
        """UPSERT records into dim_offres"""
        if not offers_data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.dim_offres (nom_offre, categorie, type_offre)
                    VALUES %s
                    ON CONFLICT (nom_offre) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                """
                
                values = [
                    (d['nom_offre'], d.get('categorie'), d.get('type_offre'))
                    for d in offers_data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Upserted {count} offer dimension records")
                return count
            except Exception as e:
                logger.error(f"Error upserting dim_offres: {e}")
                raise
            finally:
                cursor.close()
    
    def upsert_dim_geographie(self, geo_data: List[Dict]) -> int:
        """UPSERT records into dim_geographie"""
        if not geo_data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.dim_geographie
                    (city, governorate, delegation_name, locality_name, postal_code, latitude, longitude)
                    VALUES %s
                    ON CONFLICT (city, governorate, delegation_name, locality_name) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                """
                
                values = [
                    (
                        d['city'],
                        d['governorate'],
                        d.get('delegation_name'),
                        d.get('locality_name'),
                        d.get('postal_code'),
                        d.get('latitude'),
                        d.get('longitude')
                    )
                    for d in geo_data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Upserted {count} geography dimension records")
                return count
            except Exception as e:
                logger.error(f"Error upserting dim_geographie: {e}")
                raise
            finally:
                cursor.close()
    
    def upsert_dim_dealers(self, dealers_data: List[Dict]) -> int:
        """UPSERT records into dim_dealers"""
        if not dealers_data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.dim_dealers (dealer_id, dealer_name)
                    VALUES %s
                    ON CONFLICT (dealer_id) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                """
                
                values = [
                    (d['dealer_id'], d.get('dealer_name'))
                    for d in dealers_data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Upserted {count} dealer dimension records")
                return count
            except Exception as e:
                logger.error(f"Error upserting dim_dealers: {e}")
                raise
            finally:
                cursor.close()
    
    # ========== FACT TABLE OPERATIONS ==========
    
    def load_fact_abonnements(self, facts_data: List[Dict]) -> int:
        """Load fact_abonnements with foreign key mappings"""
        if not facts_data:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO etl_fibre.fact_abonnements
                    (msisdn, kit_code, date_id, offre_id, geo_id, dealer_id_pk, debit)
                    VALUES %s
                    ON CONFLICT (msisdn) DO UPDATE SET
                        kit_code = EXCLUDED.kit_code,
                        date_id = EXCLUDED.date_id,
                        offre_id = EXCLUDED.offre_id,
                        geo_id = EXCLUDED.geo_id,
                        dealer_id_pk = EXCLUDED.dealer_id_pk,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                values = [
                    (
                        f['msisdn'],
                        f['kit_code'],
                        f['date_id'],
                        f['offre_id'],
                        f['geo_id'],
                        f['dealer_id_pk'],
                        f.get('debit')
                    )
                    for f in facts_data
                ]
                
                execute_values(cursor, query, values, page_size=1000)
                count = cursor.rowcount
                logger.info(f"Loaded {count} fact records")
                return count
            except Exception as e:
                logger.error(f"Error loading facts: {e}")
                raise
            finally:
                cursor.close()
    
    # ========== LOOKUP OPERATIONS ==========
    
    def get_date_id(self, date_value) -> Optional[int]:
        """Get date_id from dim_temps"""
        query = "SELECT date_id FROM etl_fibre.dim_temps WHERE full_date = %s"
        result = self.execute_query(query, (date_value,), fetch=True)
        return result[0]['date_id'] if result else None
    
    def get_offre_id(self, offre_name: str) -> Optional[int]:
        """Get offre_id from dim_offres"""
        query = "SELECT offre_id FROM etl_fibre.dim_offres WHERE nom_offre = %s"
        result = self.execute_query(query, (offre_name,), fetch=True)
        return result[0]['offre_id'] if result else None
    
    def get_geo_id(self, city: str, governorate: str, delegation: str, locality: str) -> Optional[int]:
        """Get geo_id from dim_geographie"""
        query = """
            SELECT geo_id FROM etl_fibre.dim_geographie 
            WHERE city = %s AND governorate = %s 
            AND delegation_name = %s AND locality_name = %s
        """
        result = self.execute_query(query, (city, governorate, delegation, locality), fetch=True)
        return result[0]['geo_id'] if result else None
    
    def get_dealer_id_pk(self, dealer_id: str) -> Optional[int]:
        """Get dealer_id_pk from dim_dealers"""
        query = "SELECT dealer_id_pk FROM etl_fibre.dim_dealers WHERE dealer_id = %s"
        result = self.execute_query(query, (dealer_id,), fetch=True)
        return result[0]['dealer_id_pk'] if result else None
    
    # ========== AUDIT OPERATIONS ==========
    
    def log_etl_audit(self, audit_data: Dict) -> bool:
        """Log ETL execution audit"""
        try:
            query = """
                INSERT INTO etl_fibre.etl_audit_log
                (process_name, status, start_time, end_time, total_records_read,
                 total_records_processed, total_records_loaded, total_records_rejected, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                audit_data.get('process_name'),
                audit_data.get('status'),
                audit_data.get('start_time'),
                audit_data.get('end_time'),
                audit_data.get('total_records_read'),
                audit_data.get('total_records_processed'),
                audit_data.get('total_records_loaded'),
                audit_data.get('total_records_rejected'),
                audit_data.get('error_message')
            )
            
            self.execute_query(query, params)
            logger.info(f"Audit log recorded for {audit_data.get('process_name')}")
            return True
        except Exception as e:
            logger.error(f"Error logging audit: {e}")
            return False
    
    # ========== VALIDATION OPERATIONS ==========
    
    def validate_referential_integrity(self) -> Dict:
        """Validate foreign key relationships"""
        validations = {}
        
        try:
            # Check fact records with valid date_id
            query = """
                SELECT COUNT(*) as count 
                FROM etl_fibre.fact_abonnements 
                WHERE date_id IS NULL
            """
            result = self.execute_query(query, fetch=True)
            validations['null_date_ids'] = result[0]['count'] if result else 0
            
            # Check orphaned records
            query = """
                SELECT COUNT(*) as count 
                FROM etl_fibre.fact_abonnements f
                WHERE NOT EXISTS (SELECT 1 FROM etl_fibre.dim_temps WHERE date_id = f.date_id)
            """
            result = self.execute_query(query, fetch=True)
            validations['orphaned_dates'] = result[0]['count'] if result else 0
            
            # Check duplicate MSISDNs
            query = """
                SELECT COUNT(*)  as count 
                FROM (SELECT msisdn FROM etl_fibre.fact_abonnements GROUP BY msisdn HAVING COUNT(*) > 1) t
            """
            result = self.execute_query(query, fetch=True)
            validations['duplicate_msisdns'] = result[0]['count'] if result else 0
            
            logger.info(f"Referential integrity validation: {validations}")
            return validations
        except Exception as e:
            logger.error(f"Error validating referential integrity: {e}")
            return {"error": str(e)}

# ========== MODULE INITIALIZATION ==========

_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get or create singleton database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

if __name__ == "__main__":
    print("Database module loaded. Use get_db_manager() to access database operations.")
