#!/md
# Installation & Setup Guide

Complete step-by-step guide to set up the Fibre Data ETL Pipeline.

## 📋 Requirements

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Python:** 3.7 or higher
- **Docker:** 20.10+ with Docker Compose
- **Disk Space:** 5GB minimum
- **RAM:** 2GB minimum

## ✅ Pre-Installation Checklist

```bash
# Verify Python
python3 --version  # Should be 3.7+

# Verify Docker
docker --version
docker-compose --version

# Verify disk space
df -h | grep -E "(^/dev/|Avail)"
```

## 📦 Installation Steps

### Step 1: Clone/Navigate to Project

```bash
cd /path/to/projet-fibre-forecast
```

### Step 2: Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Edit if needed (optional - defaults work fine)
nano .env
```

Default `.env` contents:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fibre_data
DB_USER=postgres
DB_PASSWORD=postgres
```

### Step 3: Install Python Dependencies

```bash
# Option A: Using pip (recommended)
pip install psycopg2-binary pandas numpy

# Option B: Using the Makefile
make install
```

**What gets installed:**
- `psycopg2-binary`: PostgreSQL database adapter
- `pandas`: Data manipulation/analysis
- `numpy`: Numerical computing

### Step 4: Start PostgreSQL Database

```bash
# Option A: Using Docker Compose directly
cd docker
docker-compose up -d

# Option B: Using Makefile
make docker-setup
```

This starts:
- **PostgreSQL 15** container with database schema
- **PgAdmin** container (optional, at http://localhost:5050)

### Step 5: Verify Setup

```bash
# Check containers are running
docker ps

# Verify database connection
docker-compose exec postgres psql -U postgres -d fibre_data -c "SELECT version();"

# Validate configuration
cd ../src/etl
python3 config.py
```

Success output:
```
=== ETL Configuration ===

Project Root: /path/to/projet-fibre-forecast
Data Directory: /path/to/projet-fibre-forecast/data
Log File: /path/to/projet-fibre-forecast/logs/etl_pipeline_...

Database: fibre_data@localhost:5432
Required Columns: 13
Date Formats Supported: 5

Configuration Valid: True
```

## 🚀 First Run

### Step 1: Prepare Test Data

Create sample CSV file `data/landing/sample.csv`:

```csv
KIT_CODE,MSISDN,DEALER_ID,OFFRE,DEBIT,CITY,GOVERNORATE,POSTAL_CODE,LATITUDE,LONGITUDE,LOCALITY_NAME,DELEGATION_NAME,CREATION_DATE
KIT001,21698765432,M18,Pack Fibre By Ooredoo,Pack Fibre Villa 50M,El Menzah,Tunis,1001,36.8500,10.1900,El Menzah,Tunis,01/15/2024 10:30:00
KIT002,21698765433,S40,Fibre Pro,Pack Dual Play Fibre 20M,La Soukra,Ariana,2011,36.8520,10.1950,La Soukra,Ariana,01/16/2024 11:45:00
KIT003,21698765434,I37,Fibre Office In A Box,Pack Office In A Box,Ariana,Ariana,2081,36.8600,10.2000,Ariana,Ariana,01/17/2024 14:20:00
```

### Step 2: Run ETL Pipeline

```bash
# Option A: Using Makefile (recommended)
make run

# Option B: Run directly
cd src/etl
python3 etl_main.py
```

### Step 3: Monitor Execution

```bash
# In another terminal, watch the logs
make logs

# Or tail directly
tail -f logs/etl_pipeline_*.log
```

Expected output:
```
[2026-02-06 15:30:45] ==================================================
[2026-02-06 15:30:45] FIBRE DATA ETL PIPELINE
[2026-02-06 15:30:45] ==================================================
[2026-02-06 15:30:47] ✓ Extraction phase completed: 3 records
[2026-02-06 15:30:48] ✓ Transformation phase completed: 3 records
[2026-02-06 15:30:49] ✓ Loading phase completed
[2026-02-06 15:30:49] ✓ ETL PIPELINE SUCCEEDED
```

### Step 4: Verify Data Loaded

```bash
# Connect to database
docker-compose exec postgres psql -U postgres -d fibre_data

# Check data was loaded
SELECT COUNT(*) as total_subscriptions FROM etl_fibre.fact_abonnements;

# Expected output: 3 rows

# View dimensions
SELECT COUNT(*) FROM etl_fibre.dim_offres;      -- Should show 3
SELECT COUNT(*) FROM etl_fibre.dim_geographie;  -- Should show 3
SELECT COUNT(*) FROM etl_fibre.dim_dealers;     -- Should show 3

# Exit
\q
```

## 🔄 Complete Setup Using Makefile

If you prefer a single command:

```bash
make setup
```

This runs all steps in order:
1. Create `.env` from example
2. Install Python packages
3. Start Docker services
4. Wait for PostgreSQL
5. Ready to run!

Then:
```bash
# Copy your CSV files
cp your_data.csv data/landing/

# Run ETL
make run
```

## 🗂️ Directory Structure After Setup

```
projet-fibre-forecast/
├── .env                    ← Created by setup
├── data/
│   ├── landing/           ← Place CSV files here
│   ├── raw/               ← Auto-archived raw data
│   └── processed/         ← Processed CSV files
├── src/etl/
│   ├── config.py
│   ├── database.py
│   ├── extraction.py
│   ├── transformation.py
│   ├── loading.py
│   └── etl_main.py
├── docker/
│   ├── docker-compose.yml
│   └── init-scripts/
│       └── schema.sql
├── logs/                  ← Created after first run
│   ├── etl_pipeline_*.log
│   └── etl_report_*.json
└── README.md
```

## 🔧 Troubleshooting Setup

### Issue: Docker not found

```bash
# Install Docker (Ubuntu/Debian)
sudo apt-get install docker.io docker-compose

# Install Docker (macOS)
brew install docker docker-compose

# Or download Docker Desktop from https://www.docker.com/products/docker-desktop
```

### Issue: Python package installation fails

```bash
# Try with --user flag
pip install --user psycopg2-binary pandas numpy

# Or use conda if available
conda install psycopg2-binary pandas numpy

# Or Python 3 explicitly
python3 -m pip install psycopg2-binary pandas numpy
```

### Issue: PostgreSQL container won't start

```bash
# Check Docker daemon is running
docker info

# Check logs
docker logs fibre_data_postgres

# Remove old container and restart
docker-compose down -v
docker-compose up -d
```

### Issue: Port 5432 already in use

Edit `.env` and change port:
```
DB_PORT=5433
```

Then start:
```bash
docker-compose up -d
```

### Issue: Permission denied errors

```bash
# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Try again
docker ps
```

## 🧪 Testing the Installation

Run these commands to verify everything works:

```bash
# 1. Test Python environment
python3 -c "import pandas; import psycopg2; print('✓ Dependencies OK')"

# 2. Test database connection
docker-compose exec postgres psql -U postgres -d fibre_data -c "\d etl_fibre.fact_abonnements"

# 3. Test with sample data
mkdir -p data/landing
echo "KIT_CODE,MSISDN,DEALER_ID,OFFRE,DEBIT,CITY,GOVERNORATE,POSTAL_CODE,LATITUDE,LONGITUDE,LOCALITY_NAME,DELEGATION_NAME,CREATION_DATE
TEST001,21698765432,M18,Pack Fibre By Ooredoo,50M,Test City,Tunis,1001,36.85,10.19,Test,Test,01/15/2024 10:30:00" > data/landing/test.csv

# 4. Run ETL
cd src/etl && python3 etl_main.py

# 5. Verify data
docker-compose exec postgres psql -U postgres -d fibre_data -c "SELECT COUNT(*) FROM etl_fibre.fact_abonnements;"
```

## 📊 Database Access

### Via Docker CLI
```bash
docker-compose exec postgres psql -U postgres -d fibre_data
```

### Via PgAdmin Web UI
- **URL:** http://localhost:5050
- **Email:** admin@example.com (default)
- **Password:** admin (change in .env)
- Add server with `postgres` hostname

### Via GUI Tools
Configure with:
- Host: `localhost`
- Port: `5432`
- Database: `fibre_data`
- User: `postgres`
- Password: (from .env)

Compatible with: pgAdmin, DBeaver, Postico, DataGrip, etc.

## 🔄 Routine Operations

**Daily Run:**
```bash
make run
```

**Check Status:**
```bash
make logs
```

**Database Backup:**
```bash
docker-compose exec postgres pg_dump -U postgres fibre_data > backup_$(date +%Y%m%d).sql
```

**Restore Backup:**
```bash
docker-compose exec -T postgres psql -U postgres fibre_data < backup_20260206.sql
```

**Stop Services:**
```bash
make down
```

**Start Services:**
```bash
make up
```

## ✨ Next Steps

1. **Read the main README.md** for detailed documentation
2. **Prepare your CSV data** according to format specifications
3. **Configure validation rules** in `src/etl/config.py` if needed
4. **Start the watcher** to enable real-time ETL automation
5. **Create queries** to analyze the loaded data

## 📞 Connection Details

Once setup is complete:

```
PostgreSQL Database:
  - Host: localhost
  - Port: 5432
  - Database: fibre_data
  - User: postgres
  - Schema: etl_fibre

PgAdmin (optional):
  - URL: http://localhost:5050
  - Email: admin@example.com
  - Password: admin

Project Root: /path/to/projet-fibre-forecast
Data Location: /path/to/projet-fibre-forecast/data/landing/
Logs Location: /path/to/projet-fibre-forecast/logs/
```

## ✅ Installation Verification Checklist

- [ ] Python 3.7+ installed
- [ ] Docker and Docker Compose installed
- [ ] Repository cloned/project folder ready
- [ ] `.env` file created
- [ ] Python dependencies installed
- [ ] PostgreSQL container running
- [ ] Database schema initialized
- [ ] Sample CSV file created
- [ ] ETL pipeline executed successfully
- [ ] Data verified in database

🎉 **You're ready to use the ETL Pipeline!**
