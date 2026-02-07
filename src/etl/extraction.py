"""
Extraction Module - CSV Reading and Archival
Handles CSV extraction from landing directory and raw data archival
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import shutil
from config import LANDING_DIR, RAW_DIR, PROCESSED_DIR, REQUIRED_COLUMNS, CSV_COLUMN_MAPPING, ARCHIVE_CONFIG

logger = logging.getLogger(__name__)

class DataExtractor:
    """Extracts data from CSV files"""
    
    @staticmethod
    def find_csv_files() -> List[Path]:
        """Find all CSV files in landing directory"""
        try:
            csv_files = list(LANDING_DIR.glob("*.csv"))
            logger.info(f"Found {len(csv_files)} CSV file(s) in {LANDING_DIR}")
            return csv_files
        except Exception as e:
            logger.error(f"Error finding CSV files: {e}")
            return []
    
    @staticmethod
    def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that all required columns are present"""
        missing_columns = []
        df_columns_upper = [col.upper() for col in df.columns]
        
        for required_col in REQUIRED_COLUMNS:
            if required_col not in df_columns_upper:
                missing_columns.append(required_col)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False, missing_columns
        
        logger.info(f"All {len(REQUIRED_COLUMNS)} required columns found")
        return True, []
    
    @staticmethod
    def read_csv(file_path: Path) -> Optional[pd.DataFrame]:
        """Read CSV file and normalize column names"""
        try:
            logger.info(f"Reading CSV file: {file_path}")
            
            # Read CSV
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Validate columns
            is_valid, missing = DataExtractor.validate_columns(df)
            if not is_valid:
                logger.error(f"Column validation failed: {missing}")
                return None
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.strip().str.lower()
            
            # Rename columns if necessary (handle variations)
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            logger.info(f"CSV validation successful: {file_path.name}")
            return df
        
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return None
    
    @staticmethod
    def archive_raw_file(source_file: Path) -> Optional[Path]:
        """Archive the original raw CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = source_file.stem
            archived_name = f"raw_data_{filename}_{timestamp}.csv"
            archived_path = RAW_DIR / archived_name
            
            shutil.copy2(source_file, archived_path)
            logger.info(f"Raw data archived to: {archived_path}")
            return archived_path
        
        except Exception as e:
            logger.error(f"Error archiving raw file: {e}")
            return None
    
    @staticmethod
    def move_processed_file(source_file: Path) -> Optional[Path]:
        """Move processed CSV file to processed directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = source_file.stem
            processed_name = f"processed_{filename}_{timestamp}.csv"
            processed_path = PROCESSED_DIR / processed_name
            
            shutil.move(str(source_file), str(processed_path))
            logger.info(f"File moved to processed: {processed_path}")
            return processed_path
        
        except Exception as e:
            logger.error(f"Error moving processed file: {e}")
            return None
    
    @staticmethod
    def extract_to_dicts(df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of dictionaries"""
        try:
            # Fill NaN with None
            df = df.where(pd.notna(df), None)
            
            # Convert to list of dictionaries
            records = df.to_dict('records')
            logger.info(f"Converted {len(records)} rows to dictionary format")
            return records
        
        except Exception as e:
            logger.error(f"Error converting DataFrame to dicts: {e}")
            return []
    
    @staticmethod
    def get_data_stats(df: pd.DataFrame) -> Dict:
        """Generate statistics about the extracted data"""
        try:
            stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'duplicates': df.duplicated().sum(),
                'unique_msisdns': df['msisdn'].nunique() if 'msisdn' in df.columns else 0,
                'date_range': None
            }
            
            # Get date range if creation_date exists
            if 'creation_date' in df.columns:
                try:
                    dates = pd.to_datetime(df['creation_date'], errors='coerce')
                    stats['date_range'] = {
                        'min': dates.min().isoformat() if pd.notna(dates.min()) else None,
                        'max': dates.max().isoformat() if pd.notna(dates.max()) else None
                    }
                except:
                    pass
            
            logger.info(f"Data statistics: {stats['total_rows']} rows, {len(stats['missing_values'])} columns with nulls")
            return stats
        
        except Exception as e:
            logger.error(f"Error generating data statistics: {e}")
            return {}

class ExtractionPipeline:
    """Orchestrates the extraction phase"""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.extraction_log = {
            'start_time': datetime.now(),
            'files_found': 0,
            'files_processed': 0,
            'files_failed': 0,
            'total_rows_extracted': 0,
            'total_rows_rejected': 0,
            'errors': []
        }
    
    def execute(self) -> Tuple[List[Dict], Dict]:
        """Execute extraction pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("EXTRACTION PHASE STARTED")
            logger.info("=" * 60)
            
            # Find CSV files
            csv_files = self.extractor.find_csv_files()
            self.extraction_log['files_found'] = len(csv_files)
            
            if not csv_files:
                logger.warning("No CSV files found in landing directory")
                self.extraction_log['end_time'] = datetime.now()
                return [], self.extraction_log
            
            all_data = []
            
            # Process each CSV file
            for csv_file in csv_files:
                try:
                    logger.info(f"\nProcessing file: {csv_file.name}")
                    
                    # Read CSV
                    df = self.extractor.read_csv(csv_file)
                    if df is None:
                        self.extraction_log['files_failed'] += 1
                        self.extraction_log['errors'].append(f"Failed to read: {csv_file.name}")
                        continue
                    
                    # Get statistics
                    stats = self.extractor.get_data_stats(df)
                    logger.info(f"  Rows: {stats['total_rows']}, Duplicates: {stats['duplicates']}")
                    logger.info(f"  Missing values: {sum(stats['missing_values'].values())}")
                    
                    # Archive raw file
                    archived_path = self.extractor.archive_raw_file(csv_file)
                    if not archived_path:
                        self.extraction_log['errors'].append(f"Failed to archive: {csv_file.name}")
                    
                    # Convert to dictionary format
                    records = self.extractor.extract_to_dicts(df)
                    all_data.extend(records)
                    
                    # Move to processed directory
                    self.extractor.move_processed_file(csv_file)
                    
                    self.extraction_log['files_processed'] += 1
                    self.extraction_log['total_rows_extracted'] += len(records)
                    
                    logger.info(f"  ✓ Successfully processed: {csv_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing file {csv_file.name}: {e}")
                    self.extraction_log['files_failed'] += 1
                    self.extraction_log['errors'].append(f"Exception in {csv_file.name}: {str(e)}")
            
            self.extraction_log['end_time'] = datetime.now()
            
            logger.info("\n" + "=" * 60)
            logger.info("EXTRACTION PHASE COMPLETED")
            logger.info(f"Files processed: {self.extraction_log['files_processed']}/{self.extraction_log['files_found']}")
            logger.info(f"Total rows extracted: {self.extraction_log['total_rows_extracted']}")
            if self.extraction_log['errors']:
                logger.warning(f"Errors encountered: {len(self.extraction_log['errors'])}")
            logger.info("=" * 60)
            
            return all_data, self.extraction_log
        
        except Exception as e:
            logger.error(f"Critical error in extraction phase: {e}")
            self.extraction_log['end_time'] = datetime.now()
            self.extraction_log['errors'].append(f"Critical error: {str(e)}")
            return [], self.extraction_log

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    print("Extraction module loaded. Use ExtractionPipeline to extract data.")
