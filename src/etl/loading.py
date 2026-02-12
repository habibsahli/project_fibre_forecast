"""
Loading Module - Database Dimension and Fact Table Population
Handles loading dimensions and fact tables into PostgreSQL
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from calendar import monthrange
from database import DatabaseManager
from config import TIME_DIMENSION, TUNISIAN_HOLIDAYS, QUALITY_TARGETS

logger = logging.getLogger(__name__)

class DimensionBuilder:
    """Builds dimension tables"""
    
    @staticmethod
    def build_time_dimension() -> List[Dict]:
        """Generate time dimension for all dates in range"""
        try:
            dates_data = []
            
            start_year = TIME_DIMENSION['start_year']
            end_year = TIME_DIMENSION['end_year']
            
            # Generate holidays dictionary
            holidays = set(TUNISIAN_HOLIDAYS)
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    # Get last day of month
                    last_day = monthrange(year, month)[1]
                    
                    for day in range(1, last_day + 1):
                        try:
                            date_obj = datetime(year, month, day)
                            
                            dates_data.append({
                                'full_date': date_obj.date(),
                                'day_of_week': date_obj.weekday(),
                                'day_name': date_obj.strftime('%A'),
                                'week_of_year': date_obj.isocalendar()[1],
                                'month': month,
                                'month_name': date_obj.strftime('%B'),
                                'quarter': (month - 1) // 3 + 1,
                                'year': year,
                                'is_weekend': date_obj.weekday() >= 5,
                                'is_holiday': (day, month) in holidays
                            })
                        except Exception as e:
                            print(f"Error creating date {year}-{month}-{day}: {e}")
            
            print(f"Generated {len(dates_data)} time dimension records ({start_year}-{end_year})")
            return dates_data
        
        except Exception as e:
            print(f"Error building time dimension: {e}")
            return []
    
    @staticmethod
    def build_offer_dimension(records: List[Dict]) -> List[Dict]:
        """Extract unique offers and categorize them"""
        try:
            offers_dict = {}
            
            for record in records:
                if record.get('is_valid'):
                    offre = record.get('offre')
                    if offre and offre not in offers_dict:
                        offers_dict[offre] = {
                            'nom_offre': offre,
                            'categorie': record.get('offer_category', 'Other'),
                            'type_offre': 'Standard'
                        }
            
            offers_list = list(offers_dict.values())
            print(f"Generated {len(offers_list)} offer dimension records")
            return offers_list
        
        except Exception as e:
            print(f"Error building offer dimension: {e}")
            return []
    
    @staticmethod
    def build_geography_dimension(records: List[Dict]) -> List[Dict]:
        """Extract unique geographic locations"""
        try:
            geo_dict = {}
            
            for record in records:
                if record.get('is_valid'):
                    key = (
                        record.get('city'),
                        record.get('governorate'),
                        record.get('delegation_name'),
                        record.get('locality_name')
                    )
                    
                    if key[0] and key[1]:  # At least city and governorate must exist
                        if key not in geo_dict:
                            geo_dict[key] = {
                                'city': key[0],
                                'governorate': key[1],
                                'delegation_name': key[2],
                                'locality_name': key[3],
                                'postal_code': record.get('postal_code'),
                                'latitude': record.get('latitude'),
                                'longitude': record.get('longitude')
                            }
            
            geo_list = list(geo_dict.values())
            print(f"Generated {len(geo_list)} geography dimension records")
            return geo_list
        
        except Exception as e:
            print(f"Error building geography dimension: {e}")
            return []
    
    @staticmethod
    def build_dealer_dimension(records: List[Dict]) -> List[Dict]:
        """Extract unique dealers"""
        try:
            dealers_dict = {}
            
            for record in records:
                dealer_id = record.get('dealer_id')
                if dealer_id and dealer_id not in dealers_dict:
                    dealers_dict[dealer_id] = {
                        'dealer_id': dealer_id,
                        'dealer_name': f"Dealer_{dealer_id}"  # Auto-generate name
                    }
            
            dealers_list = list(dealers_dict.values())
            print(f"Generated {len(dealers_list)} dealer dimension records")
            return dealers_list
        
        except Exception as e:
            print(f"Error building dealer dimension: {e}")
            return []

class FactBuilder:
    """Builds fact tables with foreign key mappings"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.fact_records = []
        self.unmapped_records = []
    
    def build_fact_abonnements(self, records: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Build fact_abonnements with FK mappings
        Returns: (fact_records, unmapped_count)
        """
        try:
            fact_records = []
            unmapped_count = 0
            
            for record in records:
                if not record.get('is_valid'):
                    continue
                
                try:
                    # Get dimension IDs
                    date_id = self.db.get_date_id(record.get('creation_date').date()) if record.get('creation_date') else None
                    offre_id = self.db.get_offre_id(record.get('offre')) if record.get('offre') else None
                    geo_id = self.db.get_geo_id(
                        record.get('city'),
                        record.get('governorate'),
                        record.get('delegation_name'),
                        record.get('locality_name')
                    ) if record.get('city') and record.get('governorate') else None
                    dealer_id_pk = self.db.get_dealer_id_pk(record.get('dealer_id')) if record.get('dealer_id') else None
                    
                    # Check if all FKs are resolved
                    if not all([date_id, offre_id, geo_id, dealer_id_pk]):
                        unmapped_count += 1
                        logger.debug(f"Unmapped record {record.get('msisdn')}: date={date_id}, offre={offre_id}, geo={geo_id}, dealer={dealer_id_pk}")
                        continue
                    
                    fact_record = {
                        'msisdn': record.get('msisdn'),
                        'kit_code': record.get('kit_code'),
                        'date_id': date_id,
                        'offre_id': offre_id,
                        'geo_id': geo_id,
                        'dealer_id_pk': dealer_id_pk,
                        'debit': record.get('debit')
                    }
                    
                    fact_records.append(fact_record)
                
                except Exception as e:
                    logger.debug(f"Error processing record {record.get('msisdn')}: {e}")
                    unmapped_count += 1
            
            print(f"Built {len(fact_records)} fact records, {unmapped_count} unmapped")
            return fact_records, unmapped_count
        
        except Exception as e:
            print(f"Error building fact_abonnements: {e}")
            return [], len(records)

class LoadingPipeline:
    """Orchestrates the loading phase"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.dim_builder = DimensionBuilder()
        self.fact_builder = FactBuilder(db_manager)
        self.loading_log = {
            'start_time': datetime.now(),
            'status': 'IN_PROGRESS',
            'dimensions_loaded': {},
            'facts_loaded': 0,
            'validation_results': {},
            'errors': []
        }
    
    def execute(self, raw_data: List[Dict], transformed_data: List[Dict]) -> Dict:
        """Execute loading pipeline"""
        try:
            print("=" * 60)
            print("LOADING PHASE STARTED")
            print("=" * 60)
            
            # 1. LOAD RAW DATA
            print("\n[1/5] Loading raw data...")
            try:
                raw_loaded = self.db.load_raw_data(raw_data)
                print(f"  ✓ Loaded {raw_loaded} raw records")
            except Exception as e:
                print(f"  ✗ Error loading raw data: {e}")
                self.loading_log['errors'].append(f"Raw data load error: {e}")
            
            # Filter valid records for processing
            valid_records = [r for r in transformed_data if r.get('is_valid')]
            
            # 2. LOAD CLEAN DATA
            print("\n[2/5] Loading clean data...")
            try:
                valid_count, invalid_count = self.db.load_clean_data(transformed_data)
                print(f"  ✓ Loaded {valid_count} valid records, {invalid_count} invalid records")
            except Exception as e:
                print(f"  ✗ Error loading clean data: {e}")
                self.loading_log['errors'].append(f"Clean data load error: {e}")
            
            # 3. LOAD DIMENSIONS
            print("\n[3/5] Loading dimensions...")
            
            # 3a. Time Dimension
            try:
                print("  Loading dim_temps...")
                time_data = self.dim_builder.build_time_dimension()
                time_loaded = self.db.upsert_dim_temps(time_data)
                self.loading_log['dimensions_loaded']['dim_temps'] = time_loaded
                print(f"    ✓ Upserted {time_loaded} time records")
            except Exception as e:
                print(f"    ✗ Error loading dim_temps: {e}")
                self.loading_log['errors'].append(f"Time dimension error: {e}")
            
            # 3b. Offer Dimension
            try:
                print("  Loading dim_offres...")
                offer_data = self.dim_builder.build_offer_dimension(valid_records)
                offer_loaded = self.db.upsert_dim_offres(offer_data)
                self.loading_log['dimensions_loaded']['dim_offres'] = offer_loaded
                print(f"    ✓ Upserted {offer_loaded} offer records")
            except Exception as e:
                print(f"    ✗ Error loading dim_offres: {e}")
                self.loading_log['errors'].append(f"Offer dimension error: {e}")
            
            # 3c. Geography Dimension
            try:
                print("  Loading dim_geographie...")
                geo_data = self.dim_builder.build_geography_dimension(valid_records)
                geo_loaded = self.db.upsert_dim_geographie(geo_data)
                self.loading_log['dimensions_loaded']['dim_geographie'] = geo_loaded
                print(f"    ✓ Upserted {geo_loaded} geography records")
            except Exception as e:
                print(f"    ✗ Error loading dim_geographie: {e}")
                self.loading_log['errors'].append(f"Geography dimension error: {e}")
            
            # 3d. Dealer Dimension
            try:
                print("  Loading dim_dealers...")
                dealer_data = self.dim_builder.build_dealer_dimension(valid_records)
                dealer_loaded = self.db.upsert_dim_dealers(dealer_data)
                self.loading_log['dimensions_loaded']['dim_dealers'] = dealer_loaded
                print(f"    ✓ Upserted {dealer_loaded} dealer records")
            except Exception as e:
                print(f"    ✗ Error loading dim_dealers: {e}")
                self.loading_log['errors'].append(f"Dealer dimension error: {e}")
            
            # 4. LOAD FACTS
            print("\n[4/5] Loading fact_abonnements...")
            try:
                fact_records, unmapped_count = self.fact_builder.build_fact_abonnements(valid_records)
                if fact_records:
                    facts_loaded = self.db.load_fact_abonnements(fact_records)
                    self.loading_log['facts_loaded'] = facts_loaded
                    print(f"  ✓ Loaded {facts_loaded} fact records")
                    if unmapped_count > 0:
                        print(f"  ⚠ {unmapped_count} records could not be mapped to dimensions")
                else:
                    print("  No fact records to load")
            except Exception as e:
                print(f"  ✗ Error loading facts: {e}")
                self.loading_log['errors'].append(f"Fact load error: {e}")
            
            # 5. VALIDATE INTEGRITY
            print("\n[5/5] Validating referential integrity...")
            try:
                validation_results = self.db.validate_referential_integrity()
                self.loading_log['validation_results'] = validation_results
                
                if validation_results.get('null_date_ids', 0) > 0:
                    print(f"  ⚠ {validation_results['null_date_ids']} fact records with null date_ids")
                if validation_results.get('orphaned_dates', 0) > 0:
                    print(f"  ⚠ {validation_results['orphaned_dates']} orphaned date references")
                if validation_results.get('duplicate_msisdns', 0) > 0:
                    print(f"  ⚠ {validation_results['duplicate_msisdns']} duplicate MSISDNs")
                else:
                    print("  ✓ Referential integrity validated successfully")
            except Exception as e:
                print(f"  ✗ Error validating integrity: {e}")
                self.loading_log['errors'].append(f"Validation error: {e}")
            
            self.loading_log['end_time'] = datetime.now()
            self.loading_log['status'] = 'COMPLETED'
            
            print("\n" + "=" * 60)
            print("LOADING PHASE COMPLETED")
            print(f"Execution time: {(self.loading_log['end_time'] - self.loading_log['start_time']).total_seconds():.2f}s")
            print("=" * 60 + "\n")
            
            return self.loading_log
        
        except Exception as e:
            print(f"Critical error in loading phase: {e}")
            self.loading_log['end_time'] = datetime.now()
            self.loading_log['status'] = 'FAILED'
            self.loading_log['errors'].append(f"Critical error: {e}")
            return self.loading_log

if __name__ == "__main__":
    print("Loading module loaded. Use LoadingPipeline to load dimension and fact tables.")
