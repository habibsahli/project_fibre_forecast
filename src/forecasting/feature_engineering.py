"""Feature engineering utilities for forecasting models."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TUNISIAN_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-03-20", "2024-04-09", "2024-05-01",
    "2024-07-25", "2024-08-13", "2024-10-15",
    "2025-01-01", "2025-03-20", "2025-05-01",
    "2025-07-25", "2025-08-13", "2025-10-15",
])


def fill_missing_dates(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df = df.sort_values("date").reset_index(drop=True)
    date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df = df.set_index("date").reindex(date_range, fill_value=0).reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["jour_mois"] = df["date"].dt.day
    df["mois"] = df["date"].dt.month
    df["trimestre"] = df["date"].dt.quarter
    df["semaine_annee"] = df["date"].dt.isocalendar().week.astype(int)
    df["est_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
    df["jour_annee"] = df["date"].dt.dayofyear
    df["jour_annee_sin"] = np.sin(2 * np.pi * df["jour_annee"] / 365.25)
    df["jour_annee_cos"] = np.cos(2 * np.pi * df["jour_annee"] / 365.25)
    df["est_ferie"] = df["date"].isin(TUNISIAN_HOLIDAYS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["nb_abonnements"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"rolling_mean_{window}"] = df["nb_abonnements"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["nb_abonnements"].rolling(window=window).std()
    return df


def add_dimension_features(
    df: pd.DataFrame,
    df_detailed: pd.DataFrame,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df_out = df.copy()

    daily_gov = df_detailed.groupby(["date", "governorate"]).size().reset_index(name="count")
    daily_offre = df_detailed.groupby(["date", "offre_categorie"]).size().reset_index(name="count")

    top_govs = df_detailed["governorate"].value_counts().head(top_n).index.tolist()
    top_offres = df_detailed["offre_categorie"].value_counts().head(top_n).index.tolist()

    for gov in top_govs:
        gov_data = daily_gov[daily_gov["governorate"] == gov].groupby("date")["count"].sum()
        df_out = df_out.merge(
            gov_data.rename(f"gov_{gov}").to_frame().reset_index(),
            left_on="date",
            right_on="date",
            how="left",
        )
        df_out[f"gov_{gov}"] = df_out[f"gov_{gov}"].fillna(0)

    for offre in top_offres:
        offre_data = daily_offre[daily_offre["offre_categorie"] == offre].groupby("date")["count"].sum()
        df_out = df_out.merge(
            offre_data.rename(f"offre_{offre}").to_frame().reset_index(),
            left_on="date",
            right_on="date",
            how="left",
        )
        df_out[f"offre_{offre}"] = df_out[f"offre_{offre}"].fillna(0)

    return df_out, top_govs, top_offres


def build_feature_frame(
    df_daily: pd.DataFrame,
    df_detailed: pd.DataFrame,
    lags: List[int] | None = None,
    windows: List[int] | None = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    lags = lags or [1, 7, 14, 30]
    windows = windows or [7, 14, 30]

    df = fill_missing_dates(df_daily)
    df = add_time_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, windows)

    df, top_govs, top_offres = add_dimension_features(df, df_detailed, top_n=3)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["date", "nb_abonnements"]]

    return df, top_govs, top_offres
