"""Load forecasting data from PostgreSQL."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


def get_db_params() -> dict:
    load_dotenv()
    return {
        "host": os.getenv("FORECAST_DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        "port": int(os.getenv("FORECAST_DB_PORT", os.getenv("POSTGRES_PORT", 5432))),
        "database": os.getenv("FORECAST_DB_NAME", os.getenv("POSTGRES_DB", "fibre_forecast_db")),
        "user": os.getenv("FORECAST_DB_USER", os.getenv("POSTGRES_USER", "admin")),
        "password": os.getenv("FORECAST_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "SecurePassword123!")),
        "schema": os.getenv("FORECAST_DB_SCHEMA", "mart"),
    }


def get_engine():
    params = get_db_params()
    url = (
        f"postgresql://{params['user']}:{params['password']}@"
        f"{params['host']}:{params['port']}/{params['database']}"
    )
    return create_engine(url)


def load_daily_data(engine=None) -> pd.DataFrame:
    params = get_db_params()
    schema = params["schema"]
    engine = engine or get_engine()
    query = f"""
        SELECT
            DATE(f.created_at) AS date,
            COUNT(*) AS nb_abonnements
        FROM {schema}.fact_abonnements f
        GROUP BY DATE(f.created_at)
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_detailed_data(engine=None) -> pd.DataFrame:
    params = get_db_params()
    schema = params["schema"]
    engine = engine or get_engine()
    query = f"""
        SELECT
            DATE(f.created_at) AS date,
            t.jour_semaine,
            t.mois,
            t.trimestre,
            t.est_weekend,
            t.est_ferie,
            g.governorate,
            o.categorie AS offre_categorie,
            o.debit
        FROM {schema}.fact_abonnements f
        JOIN {schema}.dim_temps t ON f.date_id = t.date_id
        JOIN {schema}.dim_geographie g ON f.geo_id = g.geo_id
        JOIN {schema}.dim_offres o ON f.offre_id = o.offre_id
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def verify_daily_data(df_daily: pd.DataFrame) -> Tuple[int, str, str]:
    total = int(df_daily["nb_abonnements"].sum())
    start = df_daily["date"].min().date().isoformat()
    end = df_daily["date"].max().date().isoformat()
    return total, start, end
