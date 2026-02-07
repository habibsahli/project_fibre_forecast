-- ETL Fibre Data - Star Schema
-- PostgreSQL initialization script

-- Create schema
CREATE SCHEMA IF NOT EXISTS etl_fibre;

-- Dimension: Time
CREATE TABLE IF NOT EXISTS etl_fibre.dim_temps (
    date_id SERIAL PRIMARY KEY,
    full_date DATE UNIQUE NOT NULL,
    day_of_week INT NOT NULL,
    day_name VARCHAR(20) NOT NULL,
    week_of_year INT NOT NULL,
    month INT NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    quarter INT NOT NULL,
    year INT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Offers/Packages
CREATE TABLE IF NOT EXISTS etl_fibre.dim_offres (
    offre_id SERIAL PRIMARY KEY,
    nom_offre VARCHAR(255) UNIQUE NOT NULL,
    categorie VARCHAR(100),
    type_offre VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Geography
CREATE TABLE IF NOT EXISTS etl_fibre.dim_geographie (
    geo_id SERIAL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    governorate VARCHAR(100) NOT NULL,
    delegation_name VARCHAR(100),
    locality_name VARCHAR(100),
    postal_code INT,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, governorate, delegation_name, locality_name)
);

-- Dimension: Dealers
CREATE TABLE IF NOT EXISTS etl_fibre.dim_dealers (
    dealer_id_pk SERIAL PRIMARY KEY,
    dealer_id VARCHAR(50) UNIQUE NOT NULL,
    dealer_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact: Subscriptions
CREATE TABLE IF NOT EXISTS etl_fibre.fact_abonnements (
    abonnement_id BIGSERIAL PRIMARY KEY,
    msisdn VARCHAR(20) UNIQUE NOT NULL,
    kit_code VARCHAR(100) NOT NULL,
    date_id INT NOT NULL,
    offre_id INT NOT NULL,
    geo_id INT NOT NULL,
    dealer_id_pk INT NOT NULL,
    debit VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (date_id) REFERENCES etl_fibre.dim_temps(date_id),
    FOREIGN KEY (offre_id) REFERENCES etl_fibre.dim_offres(offre_id),
    FOREIGN KEY (geo_id) REFERENCES etl_fibre.dim_geographie(geo_id),
    FOREIGN KEY (dealer_id_pk) REFERENCES etl_fibre.dim_dealers(dealer_id_pk)
);

-- Staging/Raw Data Table
CREATE TABLE IF NOT EXISTS etl_fibre.raw_data (
    id BIGSERIAL PRIMARY KEY,
    kit_code VARCHAR(100),
    msisdn VARCHAR(20),
    dealer_id VARCHAR(50),
    offre VARCHAR(255),
    debit VARCHAR(100),
    city VARCHAR(100),
    governorate VARCHAR(100),
    postal_code INT,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    locality_name VARCHAR(100),
    delegation_name VARCHAR(100),
    creation_date TIMESTAMP,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cleaned Data Table (intermediate)
CREATE TABLE IF NOT EXISTS etl_fibre.clean_data (
    id BIGSERIAL PRIMARY KEY,
    kit_code VARCHAR(100),
    msisdn VARCHAR(20) UNIQUE,
    dealer_id VARCHAR(50),
    offre VARCHAR(255),
    debit VARCHAR(100),
    city VARCHAR(100),
    governorate VARCHAR(100),
    postal_code INT,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    locality_name VARCHAR(100),
    delegation_name VARCHAR(100),
    creation_date TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE,
    validation_errors TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit/Logging Table
CREATE TABLE IF NOT EXISTS etl_fibre.etl_audit_log (
    log_id SERIAL PRIMARY KEY,
    process_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    total_records_read INT,
    total_records_processed INT,
    total_records_loaded INT,
    total_records_rejected INT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pre-calculated Views
CREATE OR REPLACE VIEW etl_fibre.abonnements_par_jour AS
SELECT 
    dt.full_date,
    COUNT(fa.abonnement_id) as nb_abonnements
FROM etl_fibre.dim_temps dt
LEFT JOIN etl_fibre.fact_abonnements fa ON dt.date_id = fa.date_id
GROUP BY dt.full_date
ORDER BY dt.full_date DESC;

CREATE OR REPLACE VIEW etl_fibre.abonnements_par_region AS
SELECT 
    dg.governorate,
    dg.city,
    COUNT(fa.abonnement_id) as nb_abonnements
FROM etl_fibre.dim_geographie dg
LEFT JOIN etl_fibre.fact_abonnements fa ON dg.geo_id = fa.geo_id
GROUP BY dg.governorate, dg.city
ORDER BY dg.governorate, dg.city;

CREATE OR REPLACE VIEW etl_fibre.performance_dealers AS
SELECT 
    dd.dealer_id,
    COUNT(DISTINCT fa.abonnement_id) as nb_abonnements,
    COUNT(DISTINCT fa.date_id) as nb_jours_actifs,
    MIN(dt.full_date) as premiere_date,
    MAX(dt.full_date) as derniere_date
FROM etl_fibre.dim_dealers dd
LEFT JOIN etl_fibre.fact_abonnements fa ON dd.dealer_id_pk = fa.dealer_id_pk
LEFT JOIN etl_fibre.dim_temps dt ON fa.date_id = dt.date_id
GROUP BY dd.dealer_id
ORDER BY nb_abonnements DESC;

-- Create indexes for performance
CREATE INDEX idx_fact_msisdn ON etl_fibre.fact_abonnements(msisdn);
CREATE INDEX idx_fact_date_id ON etl_fibre.fact_abonnements(date_id);
CREATE INDEX idx_fact_offre_id ON etl_fibre.fact_abonnements(offre_id);
CREATE INDEX idx_fact_geo_id ON etl_fibre.fact_abonnements(geo_id);
CREATE INDEX idx_fact_dealer_id ON etl_fibre.fact_abonnements(dealer_id_pk);
CREATE INDEX idx_clean_data_msisdn ON etl_fibre.clean_data(msisdn);
CREATE INDEX idx_raw_data_msisdn ON etl_fibre.raw_data(msisdn);
CREATE INDEX idx_dim_temps_date ON etl_fibre.dim_temps(full_date);

-- Grant permissions (optional - adjust as needed)
-- GRANT SELECT ON ALL TABLES IN SCHEMA etl_fibre TO readonly_user;
-- GRANT USAGE ON SCHEMA etl_fibre TO readonly_user;
