"""
ETL Fibre Data Pipeline
========================

A complete Extract-Transform-Load pipeline for telecommunications 
fiber optic subscription data, transforming raw CSV data into 
a structured PostgreSQL database using Star Schema design.

Modules:
- config: Configuration and validation rules
- database: PostgreSQL connection and operations
- extraction: CSV extraction and archival
- transformation: Data cleaning and validation
- loading: Dimension and fact table population
- etl_main: Main ETL orchestrator

Usage:
    from src.etl.etl_main import ETLPipeline
    pipeline = ETLPipeline()
    pipeline.execute()
"""

__version__ = "1.0.0"
__author__ = "Data Engineering Team"
__date__ = "February 2026"

import logging

# Configure package logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    'config',
    'database',
    'extraction',
    'transformation',
    'loading',
    'etl_main'
]
