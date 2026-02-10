"""Forecasting pipeline for fibre subscription volume."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .data_loader import get_engine, load_daily_data, load_detailed_data
from .feature_engineering import build_feature_frame, TUNISIAN_HOLIDAYS
from .metrics import calculate_metrics, mape
from .models.prophet_model import train_prophet, predict_prophet
from .models.sarima_model import train_sarima
from .models.xgboost_model import train_xgboost
from .models.lstm_model import train_lstm, create_sequences
from .models.exp_smoothing import train_exponential_smoothing


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def ensure_output_dirs():
    for d in [MODELS_DIR, FORECASTS_DIR, REPORTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def build_prophet_holidays() -> pd.DataFrame:
    holidays = pd.DataFrame({
        "holiday": ["Holiday"] * len(TUNISIAN_HOLIDAYS),
        "ds": TUNISIAN_HOLIDAYS,
    })
    return holidays


def train_test_split(df_features: pd.DataFrame, ratio: float = 0.8):
    split_idx = int(len(df_features) * ratio)
    train = df_features.iloc[:split_idx].copy()
    test = df_features.iloc[split_idx:].copy()
    return train, test


def evaluate_models(df_daily, df_features, feature_cols, include_lstm: bool = True):
    results = {}

    # Time-series train/test
    split_idx = int(len(df_daily) * 0.8)
    train_ts = df_daily.iloc[:split_idx].copy()
    test_ts = df_daily.iloc[split_idx:].copy()

    # ML train/test
    train_df, test_df = train_test_split(df_features, ratio=0.8)
    X_train = train_df[feature_cols].values
    y_train = train_df["nb_abonnements"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["nb_abonnements"].values

    # Prophet
    try:
        start = time.time()
        df_prophet_train = pd.DataFrame({"ds": train_ts["date"], "y": train_ts["nb_abonnements"]})
        df_prophet_test = pd.DataFrame({"ds": test_ts["date"], "y": test_ts["nb_abonnements"]})
        model = train_prophet(df_prophet_train, holidays=build_prophet_holidays())
        forecast = predict_prophet(model, future_df=df_prophet_test[["ds"]])
        y_pred = forecast["yhat"].values
        metrics = calculate_metrics(df_prophet_test["y"].values, y_pred)
        metrics["Time"] = time.time() - start
        results["Prophet"] = (metrics, model)
    except Exception as exc:
        results["Prophet"] = ({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Time": 0.0}, None)

    # SARIMA
    try:
        start = time.time()
        model = train_sarima(train_ts["nb_abonnements"].values, seasonal_period=7)
        y_pred = model.predict(n_periods=len(test_ts))
        metrics = calculate_metrics(test_ts["nb_abonnements"].values, y_pred)
        metrics["Time"] = time.time() - start
        results["SARIMA"] = (metrics, model)
    except Exception:
        results["SARIMA"] = ({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Time": 0.0}, None)

    # XGBoost
    try:
        start = time.time()
        model = train_xgboost(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        metrics["Time"] = time.time() - start
        results["XGBoost"] = (metrics, model)
    except Exception:
        results["XGBoost"] = ({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Time": 0.0}, None)

    if include_lstm:
        try:
            start = time.time()
            series = train_ts["nb_abonnements"].values
            model, scaler, seq_len = train_lstm(series)
            test_series = test_ts["nb_abonnements"].values
            test_scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()
            X_seq, y_seq = create_sequences(test_scaled, seq_len)
            X_seq = X_seq.reshape(-1, seq_len, 1)
            y_pred_scaled = model.predict(X_seq, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_true = test_series[: len(y_pred)]
            metrics = calculate_metrics(y_true, y_pred)
            metrics["Time"] = time.time() - start
            results["LSTM"] = (metrics, (model, scaler, seq_len))
        except Exception:
            results["LSTM"] = ({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Time": 0.0}, None)

    # Exponential Smoothing
    try:
        start = time.time()
        name, model = train_exponential_smoothing(train_ts["nb_abonnements"].values, seasonal_period=7)
        y_pred = model.forecast(len(test_ts)) if model else np.full(len(test_ts), np.nan)
        metrics = calculate_metrics(test_ts["nb_abonnements"].values, y_pred)
        metrics["Time"] = time.time() - start
        results["Exp Smoothing"] = (metrics, model)
    except Exception:
        results["Exp Smoothing"] = ({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Time": 0.0}, None)

    return results


def select_best_model(results: dict) -> str:
    best_name = None
    best_score = float("inf")
    for name, (metrics, _) in results.items():
        score = metrics.get("MAPE", float("inf"))
        if np.isnan(score):
            continue
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def get_top_models(results: dict, top_n: int = 3) -> list[str]:
    scored = []
    for name, (metrics, _) in results.items():
        score = metrics.get("MAPE", float("inf"))
        if np.isnan(score):
            continue
        scored.append((name, score))
    scored.sort(key=lambda item: item[1])
    return [name for name, _ in scored[:top_n]]


def tune_top_models(
    top_models: list[str],
    train_ts: pd.DataFrame,
    test_ts: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_results: dict,
) -> dict:
    tuned_results = base_results.copy()

    if "Prophet" in top_models:
        best_score = float("inf")
        best_model = None
        best_metrics = None
        df_prophet_train = pd.DataFrame({"ds": train_ts["date"], "y": train_ts["nb_abonnements"]})
        df_prophet_test = pd.DataFrame({"ds": test_ts["date"], "y": test_ts["nb_abonnements"]})
        for cps in [0.01, 0.05, 0.1]:
            for sps in [0.1, 1.0, 10.0]:
                for mode in ["additive", "multiplicative"]:
                    try:
                        model = train_prophet(
                            df_prophet_train,
                            holidays=build_prophet_holidays(),
                            changepoint_prior_scale=cps,
                            seasonality_prior_scale=sps,
                            seasonality_mode=mode,
                        )
                        forecast = predict_prophet(model, future_df=df_prophet_test[["ds"]])
                        y_pred = forecast["yhat"].values
                        metrics = calculate_metrics(df_prophet_test["y"].values, y_pred)
                        if metrics["MAPE"] < best_score:
                            best_score = metrics["MAPE"]
                            best_model = model
                            best_metrics = metrics
                    except Exception:
                        continue
        if best_model is not None:
            best_metrics["Time"] = base_results["Prophet"][0].get("Time", 0.0)
            tuned_results["Prophet"] = (best_metrics, best_model)

    if "SARIMA" in top_models:
        best_score = float("inf")
        best_model = None
        best_metrics = None
        for seasonal_period in [7, 30]:
            try:
                model = train_sarima(train_ts["nb_abonnements"].values, seasonal_period=seasonal_period)
                y_pred = model.predict(n_periods=len(test_ts))
                metrics = calculate_metrics(test_ts["nb_abonnements"].values, y_pred)
                if metrics["MAPE"] < best_score:
                    best_score = metrics["MAPE"]
                    best_model = model
                    best_metrics = metrics
            except Exception:
                continue
        if best_model is not None:
            best_metrics["Time"] = base_results["SARIMA"][0].get("Time", 0.0)
            tuned_results["SARIMA"] = (best_metrics, best_model)

    if "XGBoost" in top_models:
        best_score = float("inf")
        best_model = None
        best_metrics = None
        for n_estimators in [100, 300]:
            for max_depth in [3, 5]:
                for lr in [0.05, 0.1]:
                    try:
                        model = train_xgboost(
                            X_train,
                            y_train,
                            params={
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": lr,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "random_state": 42,
                                "verbosity": 0,
                                "tree_method": "hist",
                            },
                        )
                        y_pred = model.predict(X_test)
                        metrics = calculate_metrics(y_test, y_pred)
                        if metrics["MAPE"] < best_score:
                            best_score = metrics["MAPE"]
                            best_model = model
                            best_metrics = metrics
                    except Exception:
                        continue
        if best_model is not None:
            best_metrics["Time"] = base_results["XGBoost"][0].get("Time", 0.0)
            tuned_results["XGBoost"] = (best_metrics, best_model)

    return tuned_results


def forecast_with_best_model(best_name, results, df_daily, df_features, feature_cols, top_govs, top_offres):
    last_date = df_daily["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90)

    if best_name == "Prophet":
        model = results["Prophet"][1]
        future = pd.DataFrame({"ds": future_dates})
        forecast = predict_prophet(model, future_df=future)
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
        return forecast_df

    if best_name == "SARIMA":
        model = results["SARIMA"][1]
        y_pred = model.predict(n_periods=90)
        return pd.DataFrame({"date": future_dates, "yhat": y_pred})

    if best_name == "XGBoost":
        model = results["XGBoost"][1]
        future_df = pd.DataFrame({"date": future_dates})
        future_df["jour_semaine"] = future_df["date"].dt.dayofweek
        future_df["jour_mois"] = future_df["date"].dt.day
        future_df["mois"] = future_df["date"].dt.month
        future_df["trimestre"] = future_df["date"].dt.quarter
        future_df["semaine_annee"] = future_df["date"].dt.isocalendar().week.astype(int)
        future_df["est_weekend"] = future_df["jour_semaine"].isin([5, 6]).astype(int)
        future_df["jour_annee"] = future_df["date"].dt.dayofyear
        future_df["jour_annee_sin"] = np.sin(2 * np.pi * future_df["jour_annee"] / 365.25)
        future_df["jour_annee_cos"] = np.cos(2 * np.pi * future_df["jour_annee"] / 365.25)
        future_df["est_ferie"] = future_df["date"].isin(TUNISIAN_HOLIDAYS).astype(int)

        last_known = df_features.tail(30)
        for lag in [1, 7, 14, 30]:
            future_df[f"lag_{lag}"] = last_known["nb_abonnements"].iloc[-lag]
        for window in [7, 14, 30]:
            future_df[f"rolling_mean_{window}"] = last_known["nb_abonnements"].rolling(window=window).mean().iloc[-1]
            future_df[f"rolling_std_{window}"] = last_known["nb_abonnements"].rolling(window=window).std().iloc[-1]

        for gov in top_govs:
            future_df[f"gov_{gov}"] = df_features[f"gov_{gov}"].mean()
        for offre in top_offres:
            future_df[f"offre_{offre}"] = df_features[f"offre_{offre}"].mean()

        X_future = future_df[feature_cols].values
        y_pred = model.predict(X_future)
        return pd.DataFrame({"date": future_dates, "yhat": y_pred})

    if best_name == "LSTM":
        model, scaler, seq_len = results["LSTM"][1]
        history = df_daily["nb_abonnements"].values
        history_scaled = scaler.transform(history.reshape(-1, 1)).flatten()
        input_seq = history_scaled[-seq_len:].reshape(1, seq_len, 1)

        preds = []
        for _ in range(90):
            pred = model.predict(input_seq, verbose=0)[0][0]
            preds.append(pred)
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = pred
        y_pred = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return pd.DataFrame({"date": future_dates, "yhat": y_pred})

    model = results["Exp Smoothing"][1]
    y_pred = model.forecast(90) if model else np.full(90, np.nan)
    return pd.DataFrame({"date": future_dates, "yhat": y_pred})


def save_report(comparison_df: pd.DataFrame, best_name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_comparison_report.md"
    content = ["# Model Comparison Report", "", f"Best Model: **{best_name}**", "", "## Metrics", ""]
    content.append(comparison_df.to_markdown())
    content.append("")
    content.append("## Notes")
    content.append("- Re-train monthly as new data arrives.")
    content.append("- Monitor MAPE drift and retrain if MAPE > 25%.")
    report_path.write_text("\n".join(content))
    return report_path


def save_best_model(best_name: str, results: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_obj = results[best_name][1]

    if best_name == "LSTM":
        model, _, _ = model_obj
        model_path = MODELS_DIR / "best_model.h5"
        model.save(model_path)
        return model_path

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model_obj, model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Fibre forecasting pipeline")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--no-tune", action="store_true", help="Skip lightweight tuning for top models")
    args = parser.parse_args()

    ensure_output_dirs()

    engine = get_engine()
    df_daily = load_daily_data(engine)
    df_detailed = load_detailed_data(engine)

    df_features, top_govs, top_offres = build_feature_frame(df_daily, df_detailed)
    feature_cols = [c for c in df_features.columns if c not in ["date", "nb_abonnements"]]

    results = evaluate_models(df_daily, df_features, feature_cols, include_lstm=not args.no_lstm)

    if not args.no_tune:
        split_idx = int(len(df_daily) * 0.8)
        train_ts = df_daily.iloc[:split_idx].copy()
        test_ts = df_daily.iloc[split_idx:].copy()
        train_df, test_df = train_test_split(df_features, ratio=0.8)
        X_train = train_df[feature_cols].values
        y_train = train_df["nb_abonnements"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["nb_abonnements"].values

        top_models = get_top_models(results, top_n=3)
        results = tune_top_models(
            top_models,
            train_ts,
            test_ts,
            X_train,
            y_train,
            X_test,
            y_test,
            results,
        )

    comparison_df = pd.DataFrame({k: v[0] for k, v in results.items()}).T
    best_name = select_best_model(results)

    comparison_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=True)
    save_report(comparison_df, best_name)
    model_path = save_best_model(best_name, results)

    forecast_df = forecast_with_best_model(best_name, results, df_daily, df_features, feature_cols, top_govs, top_offres)
    forecast_30 = forecast_df.head(30)
    forecast_90 = forecast_df.head(90)
    forecast_30.to_csv(FORECASTS_DIR / "forecast_30d.csv", index=False)
    forecast_90.to_csv(FORECASTS_DIR / "forecast_90d.csv", index=False)

    print(f"Best model: {best_name}")
    print(f"Forecasts saved to {FORECASTS_DIR}")
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
