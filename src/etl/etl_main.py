"""
ETL Main Orchestrator - Coordinates the complete ETL pipeline
Executes EXTRACT → TRANSFORM → LOAD workflow
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    validate_config,
    ETL_CONFIG, QUALITY_TARGETS
)
from extraction import ExtractionPipeline
from transformation import TransformationPipeline
from loading import LoadingPipeline
from database import get_db_manager

# ============= Main ETL Class =============

class ETLPipeline:
    """Main ETL Pipeline Orchestrator"""
    
    def __init__(self):
        self.db_manager = None
        self.extracted_data = []
        self.transformed_data = []
        self.execution_log = {
            'start_time': datetime.now(),
            'status': 'INITIALIZED',
            'config_valid': False,
            'extraction': {},
            'transformation': {},
            'loading': {},
            'total_execution_time': 0,
            'errors': []
        }
    
    def validate_prerequisites(self) -> bool:
        """Validate configuration and database connection"""
        print("Validating prerequisites...")
        
        # Validate configuration
        is_valid, errors = validate_config()
        if not is_valid:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            self.execution_log['errors'].extend(errors)
            return False
        
        self.execution_log['config_valid'] = True
        print("✓ Configuration valid")
        
        # Validate database connection
        try:
            self.db_manager = get_db_manager()
            print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            self.execution_log['errors'].append(f"Database error: {e}")
            return False
    
    def extract(self) -> bool:
        """Execute extraction phase"""
        try:
            print("\n" + "=" * 70)
            print("PHASE 1: EXTRACTION")
            print("=" * 70)
            
            extraction = ExtractionPipeline()
            self.extracted_data, extraction_log = extraction.execute()
            
            self.execution_log['extraction'] = extraction_log
            
            if not self.extracted_data:
                print("No data extracted. Aborting pipeline.")
                return False
            
            print(f"\n✓ Extraction phase completed: {len(self.extracted_data)} records")
            return True
        
        except Exception as e:
            print(f"Extraction phase failed: {e}")
            self.execution_log['errors'].append(f"Extraction error: {e}")
            return False
    
    def transform(self) -> bool:
        """Execute transformation phase"""
        try:
            print("\n" + "=" * 70)
            print("PHASE 2: TRANSFORMATION")
            print("=" * 70)
            
            transformation = TransformationPipeline()
            self.transformed_data, transformation_log = transformation.execute(self.extracted_data)
            
            self.execution_log['transformation'] = transformation_log
            
            if not self.transformed_data:
                print("No data after transformation. Aborting pipeline.")
                return False
            
            print(f"\n✓ Transformation phase completed: {len(self.transformed_data)} records")
            return True
        
        except Exception as e:
            print(f"Transformation phase failed: {e}")
            self.execution_log['errors'].append(f"Transformation error: {e}")
            return False
    
    def load(self) -> bool:
        """Execute loading phase"""
        try:
            print("\n" + "=" * 70)
            print("PHASE 3: LOADING")
            print("=" * 70)
            
            loading = LoadingPipeline(self.db_manager)
            loading_log = loading.execute(self.extracted_data, self.transformed_data)
            
            self.execution_log['loading'] = loading_log
            
            print(f"\n✓ Loading phase completed")
            return loading_log['status'] == 'COMPLETED'
        
        except Exception as e:
            print(f"Loading phase failed: {e}")
            self.execution_log['errors'].append(f"Loading error: {e}")
            return False
    
    def validate_quality(self) -> bool:
        """Validate data quality metrics"""
        print("\n" + "=" * 70)
        print("QUALITY VALIDATION")
        print("=" * 70)
        
        try:
            extraction_stats = self.execution_log['extraction']
            transformation_stats = self.execution_log['transformation']
            loading_stats = self.execution_log['loading']
            
            # Calculate metrics
            total_read = extraction_stats.get('total_rows_extracted', 0)
            total_processed = transformation_stats.get('total_processed', 0)
            total_valid = transformation_stats.get('total_valid', 0)
            total_loaded = loading_stats.get('facts_loaded', 0)
            
            if total_read == 0:
                print("No records to validate")
                return True
            
            # Calculate metrics
            reject_rate = (total_read - total_processed) / total_read * 100 if total_read > 0 else 0
            valid_rate = total_valid / total_processed * 100 if total_processed > 0 else 0
            load_rate = total_loaded / total_valid * 100 if total_valid > 0 else 0
            
            print(f"Records read:       {total_read}")
            print(f"Records processed:  {total_processed}")
            print(f"Valid records:      {total_valid}")
            print(f"Loaded facts:       {total_loaded}")
            print(f"\nMetrics:")
            print(f"  Reject rate:      {reject_rate:.2f}%")
            print(f"  Valid rate:       {valid_rate:.2f}%")
            print(f"  Load rate:        {load_rate:.2f}%")
            
            # Validation checks
            checks_passed = True
            
            if reject_rate > QUALITY_TARGETS['max_reject_rate_percent']:
                print(f"  ⚠ Reject rate exceeds target ({QUALITY_TARGETS['max_reject_rate_percent']}%)")
                checks_passed = False
            else:
                print(f"  ✓ Reject rate within limits")
            
            if valid_rate < 95:
                print(f"  ⚠ Valid rate below 95%")
                checks_passed = False
            else:
                print(f"  ✓ Valid rate acceptable")
            
            if load_rate < 95:
                print(f"  ⚠ Load rate below 95%")
                checks_passed = False
            else:
                print(f"  ✓ Load rate acceptable")
            
            if checks_passed:
                print("\n✓ Quality validation passed")
            else:
                print("\n⚠ Quality validation has warnings")
            
            return checks_passed
        
        except Exception as e:
            print(f"Quality validation failed: {e}")
            return False
    
    def generate_report(self):
        """Generate final execution report"""
        print("\n" + "=" * 70)
        print("EXECUTION REPORT")
        print("=" * 70)
        
        self.execution_log['end_time'] = datetime.now()
        execution_time = (self.execution_log['end_time'] - self.execution_log['start_time']).total_seconds()
        self.execution_log['total_execution_time'] = execution_time
        
        print(f"\nStatus: {self.execution_log['status']}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        if self.execution_log['errors']:
            print(f"\nErrors ({len(self.execution_log['errors'])}):")
            for error in self.execution_log['errors']:
                print(f"  - {error}")
        else:
            print("\n✓ No errors encountered")
        
        print("\n" + "=" * 70)
    
    def execute(self) -> bool:
        """Execute complete ETL pipeline"""
        print("\n" + "=" * 70)
        print("FIBRE DATA ETL PIPELINE")
        print("=" * 70)
        print(f"Start Time: {self.execution_log['start_time']}\n")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            print("Prerequisites validation failed. Aborting.")
            self.execution_log['status'] = 'FAILED'
            self.generate_report()
            return False
        
        # Execute phases
        try:
            # Phase 1: Extract
            if not self.extract():
                print("Extraction failed. Aborting.")
                self.execution_log['status'] = 'FAILED'
                self.generate_report()
                return False
            
            # Phase 2: Transform
            if not self.transform():
                print("Transformation failed. Aborting.")
                self.execution_log['status'] = 'FAILED'
                self.generate_report()
                return False
            
            # Phase 3: Load
            if not self.load():
                print("Loading failed. Aborting.")
                self.execution_log['status'] = 'FAILED'
                self.generate_report()
                return False
            
            # Validate quality
            self.validate_quality()
            
            # Mark as successful
            self.execution_log['status'] = 'SUCCESS'
            print("\n✓ ETL PIPELINE SUCCEEDED")
        
        except Exception as e:
            print(f"Unexpected error in ETL pipeline: {e}")
            self.execution_log['status'] = 'FAILED'
            self.execution_log['errors'].append(f"Unexpected error: {e}")
        
        finally:
            # Generate report
            self.generate_report()
            
            # Close database connection
            if self.db_manager:
                self.db_manager.close()
        
        return self.execution_log['status'] == 'SUCCESS'

# ============= Entry Point =============

def main():
    """Main entry point"""
    pipeline = ETLPipeline()
    success = pipeline.execute()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
