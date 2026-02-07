#!/md
# ⚡ Quick Start Guide (5 minutes)

## Prerequisites
- Docker & Docker Compose installed
- Python 3.7+
- Your CSV file ready

## 🚀 Installation

```bash
cd projet-fibre-forecast
make setup
```
**Done!** This installs everything and starts PostgreSQL.

## 📥 Load Your Data

```bash
cp your_data.csv data/landing/
```

## ▶️ Run Pipeline

```bash
make run
```

Watch logs appear in real-time. Should complete in ~30 seconds for 3,000 rows.

## ✅ Verify Results

```bash
# View logs
make logs

# Connect to database
make db-connect

# Count subscriptions
SELECT COUNT(*) FROM etl_fibre.fact_abonnements;
```

That's it! Your data is now in PostgreSQL, cleaned and organized.

---

## 🎯 Key Commands

```bash
make run              # Execute ETL
make logs             # View execution logs
make db-connect       # Access PostgreSQL
make down             # Stop database
make up               # Start database
make help             # All commands
```

---

## 📊 Sample Queries

```sql
-- Total subscriptions
SELECT COUNT(*) FROM etl_fibre.fact_abonnements;

-- By city
SELECT city, COUNT(*) 
FROM etl_fibre.dim_geographie g
JOIN etl_fibre.fact_abonnements f ON g.geo_id = f.geo_id
GROUP BY city ORDER BY COUNT(*) DESC;

-- Top dealers
SELECT dealer_id, COUNT(*) as subs
FROM etl_fibre.dim_dealers d
JOIN etl_fibre.fact_abonnements f ON d.dealer_id_pk = f.dealer_id_pk
GROUP BY dealer_id ORDER BY subs DESC;
```

---

## 📋 CSV Format

Required columns:
```
KIT_CODE, MSISDN, DEALER_ID, OFFRE, DEBIT, CITY, GOVERNORATE, 
POSTAL_CODE, LATITUDE, LONGITUDE, LOCALITY_NAME, DELEGATION_NAME, CREATION_DATE
```

Date format: `MM/DD/YYYY HH:MM:SS`

---

## 🆘 Issues?

1. **Docker error:** `make down && make up`
2. **Python error:** `pip install psycopg2-binary pandas numpy`
3. **CSV error:** Check column names (must match exactly)

---

Read [README.md](README.md) for complete documentation.

Read [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed setup.
