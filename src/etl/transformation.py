"""
Transformation Module - Data Cleaning and Validation
Handles data cleaning, normalization, and validation rules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re
from config import (
    REQUIRED_COLUMNS, CRITICAL_COLUMNS, OPTIONAL_COLUMNS,
    GEO_VALIDATION, MSISDN_VALIDATION, DATE_FORMATS, DATE_RANGE,
    QUALITY_TARGETS, categorize_offer
)

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates individual data records"""
    
    @staticmethod
    def validate_msisdn(msisdn: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate and fix Tunisian phone number (MSISDN)
        Returns: (is_valid, cleaned_msisdn, error_message)
        """
        if pd.isna(msisdn):
            return False, None, "MSISDN is null"
        
        msisdn = str(msisdn).strip()
        
        if not msisdn:
            return False, None, "MSISDN is empty"
        
        # Remove non-numeric characters
        cleaned = re.sub(r'\D', '', msisdn)
        
        # Fix Tunisian numbers
        if cleaned.startswith('216'):
            pass  # Already correct
        elif cleaned.startswith('0'):
            cleaned = '216' + cleaned[1:]
        elif len(cleaned) == 8:
            cleaned = '216' + cleaned
        
        # Validate length and format
        if len(cleaned) != MSISDN_VALIDATION['length']:
            return False, cleaned, f"Invalid length: {len(cleaned)}"
        
        if not cleaned.startswith(MSISDN_VALIDATION['required_prefix']):
            return False, cleaned, f"Missing prefix {MSISDN_VALIDATION['required_prefix']}"
        
        return True, cleaned, None
    
    @staticmethod
    def validate_date(date_str) -> Tuple[bool, Optional[datetime], Optional[str]]:
        """
        Parse and validate date string
        Returns: (is_valid, parsed_date, error_message)
        """
        if pd.isna(date_str):
            return False, None, "Date is null"
        
        date_str = str(date_str).strip()
        
        if not date_str:
            return False, None, "Date is empty"
        
        # Try multiple date formats
        for fmt in DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                
                # Validate year range
                if not (DATE_RANGE['min_year'] <= parsed_date.year <= DATE_RANGE['max_year']):
                    return False, parsed_date, f"Year out of range: {parsed_date.year}"
                
                return True, parsed_date, None
            except ValueError:
                continue
        
        return False, None, f"Could not parse date: {date_str}"
    
    @staticmethod
    def validate_coordinates(lat, lon) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """
        Validate GPS coordinates for Tunisia
        Returns: (is_valid, latitude, longitude, error_message)
        """
        try:
            if pd.isna(lat) or pd.isna(lon):
                return True, None, None, None  # Optional field, allow None
            
            lat = float(lat)
            lon = float(lon)
            
            # Check bounds for Tunisia
            if not (GEO_VALIDATION['latitude_min'] <= lat <= GEO_VALIDATION['latitude_max']):
                return True, None, None, f"Latitude out of bounds: {lat}"
            
            if not (GEO_VALIDATION['longitude_min'] <= lon <= GEO_VALIDATION['longitude_max']):
                return True, None, None, f"Longitude out of bounds: {lon}"
            
            return True, lat, lon, None
        except (ValueError, TypeError):
            return True, None, None, f"Invalid coordinate format"
    
    @staticmethod
    def validate_kit_code(kit_code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate and clean kit code"""
        if pd.isna(kit_code):
            return False, None, "KIT_CODE is null"
        
        kit_code = str(kit_code).strip().upper()
        
        if not kit_code:
            return False, None, "KIT_CODE is empty"
        
        # Reject invalid codes like "WITHOUT..."
        if kit_code.startswith('WITHOUT'):
            return False, kit_code, "Invalid kit code (starts with WITHOUT)"
        
        return True, kit_code, None
    
    @staticmethod
    def validate_text_field(value: str, field_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate and normalize text fields"""
        if pd.isna(value):
            if field_name in CRITICAL_COLUMNS:
                return False, None, f"{field_name} is null"
            return True, None, None  # Optional field
        
        text = str(value).strip()
        
        if not text:
            if field_name in CRITICAL_COLUMNS:
                return False, None, f"{field_name} is empty"
            return True, None, None
        
        # Normalize: Title case for cities, UPPER for codes
        if field_name in ['city', 'governorate', 'locality_name', 'delegation_name']:
            text = text.title()
        elif field_name in ['dealer_id', 'kit_code']:
            text = text.upper()
        
        return True, text, None

class DataTransformer:
    """Transforms and cleans data records"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.transformation_stats = {
            'total_processed': 0,
            'total_valid': 0,
            'total_invalid': 0,
            'validation_errors': {}
        }
    
    def transform_record(self, record: Dict) -> Dict:
        """Transform and validate a single record"""
        cleaned_record = record.copy()
        errors = []
        
        # 1. Validate MSISDN (critical)
        is_valid, cleaned_msisdn, error = self.validator.validate_msisdn(record.get('msisdn'))
        if not is_valid:
            errors.append(f"MSISDN: {error}")
            cleaned_record['msisdn'] = None
        else:
            cleaned_record['msisdn'] = cleaned_msisdn
        
        # 2. Validate CREATION_DATE (critical)
        is_valid, cleaned_date, error = self.validator.validate_date(record.get('creation_date'))
        if not is_valid:
            errors.append(f"CREATION_DATE: {error}")
            cleaned_record['creation_date'] = None
        else:
            cleaned_record['creation_date'] = cleaned_date
        
        # 3. Validate KIT_CODE (critical)
        is_valid, cleaned_kit, error = self.validator.validate_kit_code(record.get('kit_code'))
        if not is_valid:
            errors.append(f"KIT_CODE: {error}")
            cleaned_record['kit_code'] = None
        else:
            cleaned_record['kit_code'] = cleaned_kit
        
        # 4. Validate COORDINATES (optional but validate if present)
        is_valid, lat, lon, error = self.validator.validate_coordinates(
            record.get('latitude'),
            record.get('longitude')
        )
        if error:
            errors.append(f"COORDINATES: {error}")
        cleaned_record['latitude'] = lat
        cleaned_record['longitude'] = lon
        
        # 5. Clean TEXT FIELDS
        text_fields = ['city', 'governorate', 'locality_name', 'delegation_name', 'dealer_id', 'offre', 'debit']
        for field in text_fields:
            if field in record:
                is_valid, cleaned_text, error = self.validator.validate_text_field(record.get(field), field)
                if error:
                    errors.append(f"{field}: {error}")
                cleaned_record[field] = cleaned_text
        
        # 6. Validate POSTAL_CODE (optional)
        try:
            if pd.notna(record.get('postal_code')):
                cleaned_record['postal_code'] = int(record.get('postal_code'))
        except:
            cleaned_record['postal_code'] = None
        
        # 7. Categorize offer
        if cleaned_record.get('offre'):
            cleaned_record['offer_category'] = categorize_offer(cleaned_record['offre'])
        
        # Determine if record is valid (all critical fields present)
        has_critical_errors = any(
            error for error in errors 
            if any(critical in error for critical in list(CRITICAL_COLUMNS))
        )
        
        cleaned_record['is_valid'] = not has_critical_errors
        cleaned_record['validation_errors'] = "; ".join(errors) if errors else None
        
        return cleaned_record
    
    def transform_batch(self, records: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Transform a batch of records"""
        try:
            print("=" * 60)
            print("TRANSFORMATION PHASE STARTED")
            print(f"Processing {len(records)} records")
            print("=" * 60)
            
            transformed_records = []
            error_counts = {}
            
            for i, record in enumerate(records):
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(records)} records")
                
                transformed = self.transform_record(record)
                transformed_records.append(transformed)
                
                # Track errors
                if transformed.get('validation_errors'):
                    for error in transformed['validation_errors'].split(';'):
                        error = error.strip()
                        error_counts[error] = error_counts.get(error, 0) + 1
                
                self.transformation_stats['total_processed'] += 1
                if transformed['is_valid']:
                    self.transformation_stats['total_valid'] += 1
                else:
                    self.transformation_stats['total_invalid'] += 1
            
            # Remove duplicates (keep first occurrence by MSISDN)
            valid_records = [r for r in transformed_records if r['is_valid']]
            if valid_records:
                df_valid = pd.DataFrame(valid_records)
                df_valid = df_valid.drop_duplicates(subset=['msisdn'], keep='first')
                valid_records = df_valid.to_dict('records')
            
            invalid_records = [r for r in transformed_records if not r['is_valid']]
            
            deduped_count = self.transformation_stats['total_valid'] - len(valid_records)
            
            self.transformation_stats['validation_errors'] = error_counts
            
            print("\n" + "=" * 60)
            print("TRANSFORMATION PHASE COMPLETED")
            print(f"Total processed: {self.transformation_stats['total_processed']}")
            print(f"Valid records: {len(valid_records)}")
            print(f"Invalid records: {len(invalid_records)}")
            print(f"Duplicates removed: {deduped_count}")
            if error_counts:
                print(f"Top errors:")
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  - {error}: {count}")
            print("=" * 60 + "\n")
            
            return valid_records + invalid_records, self.transformation_stats
        
        except Exception as e:
            print(f"Critical error in transformation: {e}")
            return records, {"error": str(e)}

class TransformationPipeline:
    """Orchestrates the transformation phase"""
    
    def __init__(self):
        self.transformer = DataTransformer()
    
    def execute(self, extracted_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Execute transformation pipeline"""
        return self.transformer.transform_batch(extracted_data)

if __name__ == "__main__":
    print("Transformation module loaded. Use TransformationPipeline to transform data.")
