"""
ETL Configuration Module
Centralized configuration for the Fibre Data ETL Pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# ============= Project Paths =============
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LANDING_DIR = DATA_DIR / "landing"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
for directory in [LANDING_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============= Database Configuration =============
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "fibre_data"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "schema": "etl_fibre"
}

# ============= CSV Configuration =============
REQUIRED_COLUMNS = [
    "KIT_CODE",
    "MSISDN",
    "DEALER_ID",
    "OFFRE",
    "DEBIT",
    "CITY",
    "GOVERNORATE",
    "POSTAL_CODE",
    "LATITUDE",
    "LONGITUDE",
    "LOCALITY_NAME",
    "DELEGATION_NAME",
    "CREATION_DATE"
]

CSV_COLUMN_MAPPING = {
    "KIT_CODE": "kit_code",
    "MSISDN": "msisdn",
    "DEALER_ID": "dealer_id",
    "OFFRE": "offre",
    "DEBIT": "debit",
    "CITY": "city",
    "GOVERNORATE": "governorate",
    "POSTAL_CODE": "postal_code",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "LOCALITY_NAME": "locality_name",
    "DELEGATION_NAME": "delegation_name",
    "CREATION_DATE": "creation_date"
}

# ============= Data Validation Rules =============

# Geographic boundaries for Tunisia
GEO_VALIDATION = {
    "latitude_min": 30.0,
    "latitude_max": 38.0,
    "longitude_min": 7.0,
    "longitude_max": 12.0
}

# MSISDN Tunisian phone validation
MSISDN_VALIDATION = {
    "required_prefix": "216",
    "length": 12,  # 216 + 9 digits
}

# Date formats to try parsing
DATE_FORMATS = [
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S"
]

# Date range validation
DATE_RANGE = {
    "min_year": 2020,
    "max_year": 2026
}

# ============= Offer Categorization =============
OFFER_CATEGORIES = {
    "Pro": ["Pro", "Office", "Business", "Enterprise"],
    "Villa": ["Villa", "Residentiel", "Residential"],
    "Promo": ["Promo", "Campaign", "Promotion"]
}

def categorize_offer(offer_name: str) -> str:
    """Categorize an offer based on name patterns"""
    if not offer_name:
        return "Other"
    
    offer_upper = offer_name.upper()
    
    for category, keywords in OFFER_CATEGORIES.items():
        if any(keyword.upper() in offer_upper for keyword in keywords):
            return category
    
    return "Standard"

# ============= Critical vs Optional Columns =============
CRITICAL_COLUMNS = {
    "msisdn",
    "creation_date",
    "kit_code"
}

OPTIONAL_COLUMNS = {
    "latitude",
    "longitude",
    "locality_name",
    "delegation_name"
}

# ============= Logging Configuration =============
LOG_FILE = LOGS_DIR / f"etl_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": str(LOG_FILE),
        "console": True
    }
}

# ============= ETL Configuration =============
ETL_CONFIG = {
    "batch_size": 1000,
    "max_retries": 3,
    "timeout": 300,  # 5 minutes
    "archive_raw": True,
    "validate_foreign_keys": True,
    "max_reject_rate": 0.01  # 1% rejection rate acceptable
}

# ============= Time Dimension Configuration =============
TIME_DIMENSION = {
    "start_year": 2024,
    "end_year": 2026,
    "auto_generate": True
}

# Tunisian holidays (basic)
TUNISIAN_HOLIDAYS = [
    (1, 1),   # New Year
    (3, 20),  # Independence Day
    (4, 9),   # Martyrs' Day
    (5, 1),   # Labour Day
    (7, 25),  # Republic Day
    (8, 13),  # Women's Day
]

# ============= File Archival Configuration =============
ARCHIVE_CONFIG = {
    "naming_pattern": "raw_data_{timestamp}.csv",
    "compression": False,
    "retention_days": 90
}

# ============= Quality Metrics =============
QUALITY_TARGETS = {
    "max_execution_time_minutes": 5,
    "min_data_quality_score": 95,  # %
    "max_reject_rate_percent": 1,
    "duplicate_deduplication": True
}

# ============= Validation Rules Summary =============
def get_validation_summary() -> Dict:
    """Return a summary of all validation rules"""
    return {
        "required_columns": REQUIRED_COLUMNS,
        "critical_columns": list(CRITICAL_COLUMNS),
        "optional_columns": list(OPTIONAL_COLUMNS),
        "geo_bounds": GEO_VALIDATION,
        "msisdn_rules": MSISDN_VALIDATION,
        "date_formats": DATE_FORMATS,
        "date_range": DATE_RANGE,
        "quality_targets": QUALITY_TARGETS
    }

# ============= Helper Functions =============
def log_file_path() -> Path:
    """Get current log file path"""
    return LOG_FILE

def get_db_connection_string() -> str:
    """Generate PostgreSQL connection string"""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def validate_config() -> Tuple[bool, List[str]]:
    """Validate configuration integrity"""
    errors = []
    
    # Check critical directories exist
    for directory in [LANDING_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        if not directory.exists():
            errors.append(f"Directory missing: {directory}")
    
    # Check database configuration
    required_db_keys = ["host", "port", "database", "user", "password"]
    for key in required_db_keys:
        if not DB_CONFIG.get(key):
            errors.append(f"Database config missing: {key}")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    print("=== ETL Configuration ===\n")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Log File: {LOG_FILE}")
    print(f"\nDatabase: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"Required Columns: {len(REQUIRED_COLUMNS)}")
    print(f"Date Formats Supported: {len(DATE_FORMATS)}")
    
    is_valid, errors = validate_config()
    print(f"\nConfiguration Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
