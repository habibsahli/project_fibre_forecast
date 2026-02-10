"""Load forecasting data from PostgreSQL."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from .feature_engineering import TUNISIAN_HOLIDAYS
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
        "table": os.getenv("FORECAST_DB_TABLE", "clean_data"),
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
    table = params["table"]
    engine = engine or get_engine()
    query = f"""
        SELECT
            DATE(c.creation_date) AS date,
            COUNT(*) AS nb_abonnements
        FROM {schema}.{table} c
        WHERE c.is_valid IS NULL OR c.is_valid = TRUE
        GROUP BY DATE(c.creation_date)
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_detailed_data(engine=None) -> pd.DataFrame:
    params = get_db_params()
    schema = params["schema"]
    table = params["table"]
    engine = engine or get_engine()
    query = f"""
        SELECT
            DATE(c.creation_date) AS date,
            c.governorate,
            c.offre AS offre_raw,
            c.debit
        FROM {schema}.{table} c
        WHERE c.is_valid IS NULL OR c.is_valid = TRUE
        ORDER BY date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["mois"] = df["date"].dt.month
    df["trimestre"] = df["date"].dt.quarter
    df["est_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
    df["est_ferie"] = df["date"].isin(TUNISIAN_HOLIDAYS).astype(int)
    df["offre_categorie"] = df["offre_raw"].apply(_categorize_offer)
    df = df.drop(columns=["offre_raw"])
    return df.sort_values("date").reset_index(drop=True)


def _categorize_offer(offer_name: str) -> str:
    if not offer_name:
        return "Other"

    offer_upper = str(offer_name).upper()

    if any(keyword in offer_upper for keyword in ["PRO", "OFFICE", "BUSINESS", "ENTERPRISE"]):
        return "Pro"
    if any(keyword in offer_upper for keyword in ["VILLA", "RESIDENTIEL", "RESIDENTIAL"]):
        return "Villa"
    if any(keyword in offer_upper for keyword in ["PROMO", "CAMPAIGN", "PROMOTION"]):
        return "Promo"

    return "Standard"


def verify_daily_data(df_daily: pd.DataFrame) -> Tuple[int, str, str]:
    total = int(df_daily["nb_abonnements"].sum())
    start = df_daily["date"].min().date().isoformat()
    end = df_daily["date"].max().date().isoformat()
    return total, start, end
