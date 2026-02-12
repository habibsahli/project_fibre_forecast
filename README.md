# Fibre Data ETL Pipeline

A complete Extract-Transform-Load (ETL) pipeline for telecommunications fiber optic subscription data.

## 📋 Overview

This project transforms raw fiber optic subscription data into a structured, queryable database using a **Star Schema** design pattern.

### Architecture

```
Raw CSV Data → Extract → Transform → Load → PostgreSQL Database
                                              (Star Schema)
```

### Data Model

**Star Schema with 5 Tables:**

```
          dim_temps (dates)
              │
    ┌─────────┼─────────┬─────────┐
    │         │         │         │
dim_offres  dim_geo  dim_dealers  fact_abonnements
(packages) (cities) (vendors)     (subscriptions)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Docker & Docker Compose
- PostgreSQL 13+ (via Docker)
- 2GB disk space minimum

### Installation

1. **Clone/Setup Project**
   ```bash
   cd projet-fibre-forecast
   ```

2. **Configure Environment**
   ```bash
   make env-setup
   # Edit .env with your database credentials if needed
   cat .env
   ```

3. **Install & Start**
   ```bash
   make setup  # Installs dependencies + starts Docker
   ```

4. **Verify Setup**
   ```bash
   make validate  # Checks configuration
   ```

### Running the Pipeline

1. **Prepare Data**
   ```bash
   # Copy your CSV files to:
   cp your_data.csv data/landing/
   ```

2. **Execute Pipeline**
   ```bash
   make run
   ```

3. **Run Forecasting (Phase 2)**
   ```bash
   make forecast
   # or
   ./.venv/bin/python forecast_cli.py run
   # skip LSTM if needed
   ./.venv/bin/python forecast_cli.py run --no-lstm
   ```

3. **Auto-run on new files (recommended for continuous ingestion)**
   ```bash
   # Install watcher dependency (Linux)
   sudo apt-get install inotify-tools

   # Start the watcher
   make watch
   ```
   The watcher monitors `data/landing/` and triggers the ETL whenever a new CSV
   file is added or moved into the folder.

   To debounce rapid file drops (default 10 seconds):
   ```bash
   DEBOUNCE_SECONDS=15 make watch
   ```

   Email notification on launch (requires `mail` or `sendmail` installed):
   ```bash
   NOTIFY_EMAIL=habib.sahli@esprit.tn make watch
   ```
   
   **Email Setup Options:**
   
   1. **Local Mail Server (Recommended for servers):**
      ```bash
      sudo apt-get install postfix mailutils
      NOTIFY_EMAIL=your.email@domain.com make watch
      ```
   
   2. **External SMTP (Gmail, Outlook, etc):**
      ```bash
      export SMTP_SERVER=smtp.gmail.com
      export SMTP_PORT=587
      export SMTP_USER=your.email@gmail.com
      export SMTP_PASSWORD=your_app_password
      export NOTIFY_EMAIL=recipient@domain.com
      make watch
      ```
   
   You can also customize sender and subject prefix:
   ```bash
   NOTIFY_EMAIL=habib.sahli@esprit.tn \
   FROM_EMAIL=ETL-notify@local \
   SUBJECT_PREFIX="ETL Launch" \
   make watch
   ```

3. **Check Results**
   ```bash
   make logs  # View execution logs
   make db-connect  # Access database
   ```

   ## 📈 Forecasting Pipeline Flow (Phase 2)

   ```mermaid
   flowchart TD
      A[Load clean data from PostgreSQL] --> B[Fill missing dates]
      B --> C[Feature engineering
      - time features
      - lags
      - rolling means]
      C --> D[Temporal train/test split]
      D --> E[Baseline run for all models]
      E --> F[Pick top 2-3 models by MAPE]
      F --> G[Lightweight tuning]
      G --> H[Select best model]
      H --> I[Train on full data]
      I --> J[Forecast 30/90 days]
      J --> K[Save forecasts + report + model]
   ```

## 📁 Project Structure

```
projet-fibre-forecast/
├── data/
│   ├── landing/       ← DROP CSV FILES HERE
│   ├── raw/           ← Auto-archived raw data
│   └── processed/     ← Processed CSV files
├── src/etl/
│   ├── config.py           (Configuration & validation rules)
│   ├── database.py         (PostgreSQL operations)
│   ├── extraction.py       (Read & archive CSV)
│   ├── transformation.py   (Clean & validate data)
│   ├── loading.py          (Load dimensions & facts)
│   └── etl_main.py         (Main orchestrator)
├── docker/
│   ├── docker-compose.yml  (PostgreSQL + PgAdmin)
│   └── init-scripts/
│       └── schema.sql      (Database schema)
├── Makefile                (Common commands)
└── .env.example            (Configuration template)
```

## 🔄 ETL Phases

### Phase 1: EXTRACT
- Scans `data/landing/` for CSV files
- Validates required columns
- Archives raw data to `data/raw/`
- Moves processed files to `data/processed/`

**Output:** Raw data records in memory

### Phase 2: TRANSFORM
- Validates MSISDN (phone numbers) - fixes Tunisian format
- Parses and validates dates (multiple formats supported)
- Normalizes text fields (Title case for cities)
- Validates GPS coordinates (bounds check for Tunisia)
- Removes duplicates (keeps first occurrence)
- Tracks validation errors for audit

**Output:** Cleaned & validated records

### Phase 3: LOAD
1. **Archive Raw Data** → `etl_fibre.raw_data` table
2. **Load Clean Data** → `etl_fibre.clean_data` table
3. **Populate Dimensions:**
   - `dim_temps`: 1,096 dates (2024-2026)
   - `dim_offres`: Unique packages (auto-categorized)
   - `dim_geographie`: Unique locations (city + governorate)
   - `dim_dealers`: Unique dealers/vendors
4. **Load Facts** → `fact_abonnements` (main fact table)
5. **Validate Integrity** → FK relationships checked

## 📊 Database Schema

### Dimension Tables

#### `dim_temps` (Time)
- `date_id`: Primary key
- `full_date`: Date value
- `day_of_week`, `day_name`: Day information
- `month`, `month_name`, `quarter`, `year`: Date components
- `is_weekend`, `is_holiday`: Boolean flags

#### `dim_offres` (Offers/Packages)
- `offre_id`: Primary key
- `nom_offre`: Package name (unique)
- `categorie`: Auto-categorized (Pro, Villa, Promo, Standard)
- `type_offre`: Package type

#### `dim_geographie` (Geography)
- `geo_id`: Primary key
- `city`, `governorate`, `delegation_name`, `locality_name`
- `postal_code`, `latitude`, `longitude`
- **Unique Index:** (city, governorate, delegation_name, locality_name)

#### `dim_dealers` (Vendors)
- `dealer_id_pk`: Primary key
- `dealer_id`: Dealer code (unique)
- `dealer_name`: Dealer name

### Fact Table

#### `fact_abonnements` (Subscriptions)
- `abonnement_id`: Primary key
- `msisdn`: Phone number (unique)
- `kit_code`: Equipment code
- `date_id`: FK → `dim_temps`
- `offre_id`: FK → `dim_offres`
- `geo_id`: FK → `dim_geographie`
- `dealer_id_pk`: FK → `dim_dealers`
- `debit`: Speed/package details

### Supporting Tables

- `raw_data`: Original CSV data (archived as-is)
- `clean_data`: Cleaned data with validation audit
- `etl_audit_log`: Execution history & statistics

## 🔍 Data Validation Rules

### Critical Fields (reject record if invalid)
- **MSISDN**: Must be valid Tunisian format (starts with 216, 12 digits total)
- **CREATION_DATE**: Must parse to valid date (2020-2026)
- **KIT_CODE**: Cannot be empty or start with "WITHOUT"

### Optional Fields (NULL allowed)
- LATITUDE, LONGITUDE: Bounds-checked if present (Tunisia: 30-38°N, 7-12°E)
- LOCALITY_NAME, DELEGATION_NAME

### Automatic Fixes
- MSISDN: Auto-converts `0xxxxxxxxx` → `216xxxxxxxxx`
- Dates: Tries multiple formats
- Text: Normalizes whitespace, case

### Duplicate Handling
- MSISDN duplicates: Keep first occurrence
- Exact duplicates: Removed
- UPSERT behavior: Updates if MSISDN already exists

## 📈 Pre-Calculated Views

Query these views for quick analysis:

```sql
-- Subscriptions by day
SELECT * FROM etl_fibre.abonnements_par_jour;

-- Subscriptions by region
SELECT * FROM etl_fibre.abonnements_par_region;

-- Dealer performance
SELECT * FROM etl_fibre.performance_dealers;
```

## 🛠️ Common Commands

```bash
# Quick commands
make up              # Start database
make run             # Execute ETL
make logs            # View logs
make down            # Stop services

# Database
make db-connect      # Access PostgreSQL CLI
make db-reset        # Clear all data (WARNING!)

# Development
make validate        # Check config
make lint            # Check Python syntax
make test            # Create test data directory
```

## 🔐 Configuration

Edit `.env` file to customize:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fibre_data
DB_USER=postgres
DB_PASSWORD=postgres
```

### Validation Rules (config.py)

Modify to adjust:
- Geographic bounds (`GEO_VALIDATION`)
- Date formats (`DATE_FORMATS`)
- Offer categories (`OFFER_CATEGORIES`)
- Quality targets (`QUALITY_TARGETS`)

## 📊 Sample Queries

```sql
-- Total subscriptions
SELECT COUNT(*) FROM etl_fibre.fact_abonnements;

-- Subscriptions by governorate
SELECT dg.governorate, COUNT(*) as count
FROM etl_fibre.fact_abonnements fa
JOIN etl_fibre.dim_geographie dg ON fa.geo_id = dg.geo_id
GROUP BY dg.governorate
ORDER BY count DESC;

-- Top packages
SELECT do.nom_offre, COUNT(*) as count
FROM etl_fibre.fact_abonnements fa
JOIN etl_fibre.dim_offres do ON fa.offre_id = do.offre_id
GROUP BY do.nom_offre
ORDER BY count DESC;

-- Dealer performance
SELECT dd.dealer_id, COUNT(*) as subs
FROM etl_fibre.fact_abonnements fa
JOIN etl_fibre.dim_dealers dd ON fa.dealer_id_pk = dd.dealer_id_pk
GROUP BY dd.dealer_id
ORDER BY subs DESC;

-- Subscriptions last 7 days
SELECT COUNT(*)
FROM etl_fibre.fact_abonnements fa
JOIN etl_fibre.dim_temps dt ON fa.date_id = dt.date_id
WHERE dt.full_date >= CURRENT_DATE - INTERVAL '7 days';
```

## 🔔 Real-Time Automation

The ETL pipeline automatically triggers when new CSV files are added to `data/landing/` via the **file watcher**.

**The watcher (`watch_etl.sh`) is active and handles all ETL automation.** No cron job is needed.

### Watcher Features:
1. Continuously monitors `data/landing/` for new CSV files
2. Triggers ETL immediately upon file detection
3. 10-second debounce window to batch multiple file uploads
4. Sends email notifications on launch (if configured)
5. Prevents concurrent ETL runs with file locking
6. Auto-starts PostgreSQL container if needed

### Start the Watcher:
```bash
# Via Makefile
make watch

# Or directly
bash watch_etl.sh
```

## 📋 CSV Format Requirements

**Required Columns:**
- `KIT_CODE`: Equipment identifier
- `MSISDN`: Phone number (unique key)
- `DEALER_ID`: Vendor code
- `OFFRE`: Package name
- `DEBIT`: Speed/plan details
- `CITY`: City name
- `GOVERNORATE`: Province/region
- `POSTAL_CODE`: Zip code
- `LATITUDE`: GPS latitude
- `LONGITUDE`: GPS longitude
- `LOCALITY_NAME`: Neighborhood
- `DELEGATION_NAME`: District
- `CREATION_DATE`: Subscription date (MM/DD/YYYY HH:MM:SS)

**Example CSV:**
```
KIT_CODE,MSISDN,DEALER_ID,OFFRE,DEBIT,CITY,GOVERNORATE,POSTAL_CODE,LATITUDE,LONGITUDE,LOCALITY_NAME,DELEGATION_NAME,CREATION_DATE
KIT001,21698765432,M18,Pack Fibre By Ooredoo,Pack Fibre Villa 50M,El Menzah,Tunis,1001,36.8500,10.1900,El Menzah,Tunis,01/15/2024 10:30:00
```

## ✅ Quality Validation

The pipeline validates:

1. **Data Completeness:**
   - All critical columns present
   - No null MSISDNs in facts
   - Proper foreign keys

2. **Data Accuracy:**
   - Valid dates (2020-2026)
   - Valid phone numbers
   - Valid coordinates (Tunisia bounds)

3. **Data Consistency:**
   - No orphaned references
   - No duplicate MSISDNs in facts
   - Integrity constraints pass

4. **Performance:**
   - Processing < 5 minutes
   - Reject rate < 1%
   - Load success rate > 95%

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart PostgreSQL
make down && make up
```

### CSV File Not Processing
```bash
# Verify file is in correct location
ls data/landing/

# Check for column name issues (must match required columns)
head -1 data/landing/your_file.csv
```

### Data Validation Failures
```bash
# Verify date format matches (MM/DD/YYYY HH:MM:SS)
# Verify MSISDN is valid Tunisian number
```

### Database Storage Full
```bash
# Check disk space
df -h

# Clean old logs
make clean  # Be careful with this!
```

## 📈 Performance

**Typical Execution Times:**
- 3,000 records: ~30 seconds
- 10,000 records: ~1.5 minutes
- 100,000 records: ~10 minutes

**Database Size:**
- ~1 KB per subscription record
- 3,000 records ≈ 3 MB
- Indexes add ~10% overhead

## 🔐 Security

- Credentials in `.env` (never commit)
- SQL injection prevention via parameterized queries
- FK constraints prevent orphaned data
- Audit logging of all operations
- Raw data archival for traceability

## 🚀 Future Enhancements

**Phase 2: Forecasting**
- Time series analysis (Prophet)
- Demand forecasts

**Phase 3: Dashboard**
- Interactive BI dashboards
- Geographic heatmaps
- Real-time metrics

**Phase 4: AI**
- Anomaly detection
- Customer segmentation
- Predictive churn modeling

## 📞 Support

For issues:
1. Check logs: `make logs`
2. Verify configuration: `make validate`
3. Test database: `make db-connect`

## 📜 License

Internal Use Only - Ooredoo Tunisia

---

**Last Updated:** February 2026
**Version:** 1.0
