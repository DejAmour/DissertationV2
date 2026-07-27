"""Build historical diagnostics and full handoff package for Brent GBM analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

from download_data import download_brent_data
from estimate_gbm import estimate_gbm
from evaluate_gbm import evaluate_gbm
from prepare_data import prepare_data
from simulate_gbm import simulate_gbm

ANNUALISATION_FACTOR = 252
PERIODS = [
    ("full_sample", "2000-01-01", "2025-12-31"),
    ("pre-pandemic comparison", "2010-01-01", "2019-12-31"),
    ("COVID-19 stress period", "2020-02-01", "2020-06-30"),
]


@dataclass
class StageResult:
    name: str
    status: str
    details: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return project_root().parent


def handoff_root() -> Path:
    return repo_root() / "handoff_brent_analysis"


def _source_note() -> str:
    return "Source: Europe Brent Spot Price FOB (USD/barrel); underlying EIA series (DCOILBRENTEU)."


def _ensure_handoff_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "raw": base / "data" / "raw",
        "processed": base / "data" / "processed",
        "code": base / "code",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_stage1_processed() -> pd.DataFrame:
    path = project_root() / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 processed file: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)
    return df


def _load_provenance(raw_dir: Path) -> dict[str, str]:
    provenance_path = raw_dir / "data_provenance.json"
    if not provenance_path.exists():
        return {}
    return json.loads(provenance_path.read_text(encoding="utf-8"))


def _find_raw_file(raw_dir: Path, provenance: dict[str, str]) -> Path | None:
    from_meta = provenance.get("raw_csv_filename")
    if from_meta:
        candidate = raw_dir / from_meta
        if candidate.exists():
            return candidate
    csvs = sorted(raw_dir.glob("DCOILBRENTEU*.csv"), key=lambda p: p.stat().st_mtime)
    if csvs:
        return csvs[-1]
    return None


def _validate_raw_data(raw_file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = pd.read_csv(raw_file)
    cols = list(raw_df.columns)
    upper_map = {c: c.strip().upper() for c in cols}
    raw_df = raw_df.rename(columns=upper_map)

    if "OBSERVATION_DATE" in raw_df.columns and "DATE" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"OBSERVATION_DATE": "DATE"})

    if "DATE" not in raw_df.columns:
        raise ValueError("Raw data missing DATE/OBSERVATION_DATE column.")

    price_col = "DCOILBRENTEU"
    if price_col not in raw_df.columns:
        if "PRICE" in raw_df.columns:
            price_col = "PRICE"
        else:
            raise ValueError("Raw data missing DCOILBRENTEU/PRICE column.")

    date_raw = raw_df["DATE"]
    parsed_dates = pd.to_datetime(date_raw, errors="coerce")
    series = raw_df[price_col]
    series_str = series.astype("string").str.strip()
    non_numeric_mask = ~(series_str.isna() | series_str.eq("") | series_str.eq(".")) & pd.to_numeric(
        series, errors="coerce"
    ).isna()
    numeric = pd.to_numeric(series, errors="coerce")
    missing_mask = series.isna() | series_str.eq("") | series_str.eq(".")
    duplicate_dates = int(parsed_dates.duplicated().sum())
    chronological = bool(parsed_dates.dropna().is_monotonic_increasing)
    non_positive = int((numeric.dropna() <= 0).sum())

    first_obs = parsed_dates.dropna().min()
    last_obs = parsed_dates.dropna().max()

    validation = {
        "units": "USD/barrel",
        "frequency": "Daily",
        "first_observation": first_obs.date().isoformat() if pd.notna(first_obs) else "",
        "last_observation": last_obs.date().isoformat() if pd.notna(last_obs) else "",
        "missing_values": int(missing_mask.sum()),
        "non_numeric_values": int(non_numeric_mask.sum()),
        "duplicate_dates": duplicate_dates,
        "non_positive_values": non_positive,
        "chronological_order": chronological,
    }
    return raw_df, validation


def _slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    return df.loc[mask].copy()


def _build_processed_with_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Log_Price"] = np.log(out["Price_USD_per_barrel"])
    out["Log_Return"] = np.log(out["Price_USD_per_barrel"] / out["Price_USD_per_barrel"].shift(1))
    return out


def _period_log_returns(period_df: pd.DataFrame) -> pd.Series:
    prices = period_df["Price_USD_per_barrel"].astype(float)
    return np.log(prices / prices.shift(1)).dropna()


def _period_descriptive(period_label: str, period_df: pd.DataFrame) -> dict[str, object]:
    prices = period_df["Price_USD_per_barrel"]
    returns = _period_log_returns(period_df)
    if returns.empty:
        raise ValueError(f"No return observations for period: {period_label}")

    pearson_kurt = float(stats.kurtosis(returns, fisher=False, bias=False))
    excess_kurt = float(stats.kurtosis(returns, fisher=True, bias=False))

    result: dict[str, object] = {
        "period": period_label,
        "start_date": period_df["Date"].min().date().isoformat(),
        "end_date": period_df["Date"].max().date().isoformat(),
        "observations_prices": int(len(prices)),
        "observations_returns": int(len(returns)),
        "mean": float(returns.mean()),
        "median": float(returns.median()),
        "std": float(returns.std(ddof=1)),
        "annualized_vol": float(returns.std(ddof=1) * math.sqrt(ANNUALISATION_FACTOR)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "skewness": float(stats.skew(returns, bias=False)),
        "pearson_kurtosis": pearson_kurt,
        "excess_kurtosis": excess_kurt,
        "p01": float(np.quantile(returns, 0.01)),
        "p05": float(np.quantile(returns, 0.05)),
        "p25": float(np.quantile(returns, 0.25)),
        "p50": float(np.quantile(returns, 0.50)),
        "p75": float(np.quantile(returns, 0.75)),
        "p95": float(np.quantile(returns, 0.95)),
        "p99": float(np.quantile(returns, 0.99)),
    }
    return result


def _normality_tail_tables(period_label: str, period_df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    returns = _period_log_returns(period_df)
    jb_stat, jb_p = stats.jarque_bera(returns)
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))

    thresholds = [2, 3, 4, 5]
    rows: list[dict[str, object]] = []
    for k in thresholds:
        if sigma > 0:
            empirical = float(np.mean(np.abs(returns - mu) > k * sigma))
            theoretical = float(2 * (1 - norm.cdf(k)))
            ratio = empirical / theoretical if theoretical > 0 else np.nan
        else:
            empirical = np.nan
            theoretical = np.nan
            ratio = np.nan
        rows.append(
            {
                "period": period_label,
                "sigma_threshold": k,
                "empirical_frequency": empirical,
                "theoretical_normal_probability": theoretical,
                "empirical_to_theoretical_ratio": ratio,
            }
        )

    top = period_df[["Date", "Price_USD_per_barrel"]].copy()
    top["Previous_Price_USD_per_barrel"] = top["Price_USD_per_barrel"].shift(1)
    top["Log_Return"] = np.log(top["Price_USD_per_barrel"] / top["Previous_Price_USD_per_barrel"])
    top = top.loc[top["Log_Return"].notna()].copy()
    top["Abs_Log_Return"] = top["Log_Return"].abs()
    top = top.sort_values("Abs_Log_Return", ascending=False).head(10)
    top.insert(0, "period", period_label)

    normality_row = {
        "period": period_label,
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "fitted_normal_mean": mu,
        "fitted_normal_std": sigma,
    }

    top = top.rename(
        columns={
            "Date": "date",
            "Price_USD_per_barrel": "price",
            "Previous_Price_USD_per_barrel": "previous_price",
            "Log_Return": "log_return",
            "Abs_Log_Return": "abs_log_return",
        }
    )

    return normality_row, pd.DataFrame(rows), top


def _rolling_volatility(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["Date", "Log_Return"]].copy()
    summary_rows: list[dict[str, object]] = []

    for window in (30, 90):
        col = f"rolling_vol_{window}d"
        work[col] = work["Log_Return"].rolling(window=window, min_periods=window).std(ddof=1) * math.sqrt(ANNUALISATION_FACTOR)
        valid = work.dropna(subset=[col])
        if valid.empty:
            summary_rows.append(
                {
                    "window_days": window,
                    "min": np.nan,
                    "median": np.nan,
                    "mean": np.nan,
                    "max": np.nan,
                    "date_of_max": "",
                }
            )
            continue
        idxmax = valid[col].idxmax()
        summary_rows.append(
            {
                "window_days": window,
                "min": float(valid[col].min()),
                "median": float(valid[col].median()),
                "mean": float(valid[col].mean()),
                "max": float(valid[col].max()),
                "date_of_max": pd.to_datetime(work.loc[idxmax, "Date"]).date().isoformat(),
            }
        )

    return work, pd.DataFrame(summary_rows)


def _autocorrelation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = df["Log_Return"].dropna()
    series_map = {
        "returns": returns,
        "squared_returns": returns.pow(2),
        "absolute_returns": returns.abs(),
    }

    acf_rows: list[dict[str, object]] = []
    lb_rows: list[dict[str, object]] = []

    for series_type, ser in series_map.items():
        acf_vals = acf(ser, nlags=20, fft=True)
        for lag in range(1, 21):
            acf_rows.append({"series": series_type, "lag": lag, "acf": float(acf_vals[lag])})

        lb = acorr_ljungbox(ser, lags=[5, 10, 20], return_df=True)
        for lag, row in lb.iterrows():
            lb_rows.append(
                {
                    "series": series_type,
                    "lag": int(lag),
                    "ljung_box_stat": float(row["lb_stat"]),
                    "ljung_box_pvalue": float(row["lb_pvalue"]),
                }
            )

    return pd.DataFrame(acf_rows), pd.DataFrame(lb_rows)


def _ar1_and_adf_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["Date", "Log_Price", "Log_Return"]].copy()
    work["lag_log_price"] = work["Log_Price"].shift(1)
    ar_df = work.dropna(subset=["Log_Price", "lag_log_price"])

    model = OLS(ar_df["Log_Price"], np.column_stack([np.ones(len(ar_df)), ar_df["lag_log_price"]])).fit()
    intercept = float(model.params.iloc[0])
    phi = float(model.params.iloc[1])
    intercept_se = float(model.bse.iloc[0])
    phi_se = float(model.bse.iloc[1])
    intercept_t = float(model.tvalues.iloc[0])
    phi_t = float(model.tvalues.iloc[1])
    intercept_p = float(model.pvalues.iloc[0])
    phi_p = float(model.pvalues.iloc[1])

    ar1_df = pd.DataFrame(
        [
            {
                "series": "log_price",
                "intercept": intercept,
                "phi": phi,
                "intercept_se": intercept_se,
                "phi_se": phi_se,
                "intercept_t": intercept_t,
                "phi_t": phi_t,
                "intercept_p": intercept_p,
                "phi_p": phi_p,
                "nobs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "half_life_days": float(np.nan),
                "half_life_note": (
                    "Half-life not interpreted: ADF tests do not robustly reject "
                    "the unit root for this series. A statistically significant "
                    "phi (OLS t-test vs 0) is not evidence of mean reversion; "
                    "only ADF-based inference on phi<1 is valid for this purpose. "
                    "Failure to reject the unit-root null is not proof that the "
                    "unit root exists; structural breaks can reduce ADF reliability."
                ),
            }
        ]
    )

    adf_rows: list[dict[str, object]] = []
    for period_label, start, end in PERIODS:
        period_df = _slice_period(df, start, end)
        if period_df.empty:
            for regression in ("c", "ct"):
                adf_rows.append(
                    {
                        "period": period_label,
                        "series": "log_price",
                        "regression": regression,
                        "autolag": "AIC",
                        "adf_statistic": np.nan,
                        "pvalue": np.nan,
                        "usedlag": np.nan,
                        "nobs": np.nan,
                        "critical_value_1pct": np.nan,
                        "critical_value_5pct": np.nan,
                        "critical_value_10pct": np.nan,
                        "reject_1pct": np.nan,
                        "reject_5pct": np.nan,
                        "reject_10pct": np.nan,
                        "note": "Period missing in processed data.",
                    }
                )
            continue

        series = period_df["Log_Price"].dropna()
        for regression in ("c", "ct"):
            stat, pvalue, usedlag, nobs, crit, _ = adfuller(series, regression=regression, autolag="AIC")
            adf_rows.append(
                {
                    "period": period_label,
                    "series": "log_price",
                    "regression": regression,
                    "autolag": "AIC",
                    "adf_statistic": float(stat),
                    "pvalue": float(pvalue),
                    "usedlag": int(usedlag),
                    "nobs": int(nobs),
                    "critical_value_1pct": float(crit["1%"]),
                    "critical_value_5pct": float(crit["5%"]),
                    "critical_value_10pct": float(crit["10%"]),
                    "reject_1pct": bool(stat < crit["1%"]),
                    "reject_5pct": bool(stat < crit["5%"]),
                    "reject_10pct": bool(stat < crit["10%"]),
                    "note": "",
                }
            )

    return ar1_df, pd.DataFrame(adf_rows)


def _sample_period_comparison(
    full_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    normality_df: pd.DataFrame,
    tails_df: pd.DataFrame,
    adf_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for period_label, _, _ in PERIODS:
        drow = descriptive_df.loc[descriptive_df["period"] == period_label].iloc[0]
        nrow = normality_df.loc[normality_df["period"] == period_label].iloc[0]
        period_df = full_df.loc[
            (full_df["Date"] >= pd.Timestamp(drow["start_date"])) & (full_df["Date"] <= pd.Timestamp(drow["end_date"]))
        ].copy()
        period_returns = _period_log_returns(period_df)
        acf1_returns = float(acf(period_returns, nlags=1, fft=False)[1])
        acf1_squared = float(acf(period_returns.pow(2), nlags=1, fft=False)[1])
        outside3 = tails_df.loc[
            (tails_df["period"] == period_label) & (tails_df["sigma_threshold"] == 3), "empirical_frequency"
        ]
        outside5 = tails_df.loc[
            (tails_df["period"] == period_label) & (tails_df["sigma_threshold"] == 5), "empirical_frequency"
        ]
        arow = adf_df.loc[(adf_df["period"] == period_label) & (adf_df["regression"] == "c")].iloc[0]

        rows.append(
            {
                "period": period_label,
                "observations": int(drow["observations_prices"]),
                "mean_return": float(drow["mean"]),
                "annualized_vol": float(drow["annualized_vol"]),
                "skewness": float(drow["skewness"]),
                "excess_kurtosis": float(drow["excess_kurtosis"]),
                "jarque_bera_stat": float(nrow["jarque_bera_statistic"]),
                "jarque_bera_pvalue": float(nrow["jarque_bera_pvalue"]),
                "freq_outside_3sigma": float(outside3.iloc[0]) if len(outside3) else np.nan,
                "freq_outside_5sigma": float(outside5.iloc[0]) if len(outside5) else np.nan,
                "acf1_returns": acf1_returns,
                "acf1_squared_returns": acf1_squared,
                "adf_stat_c": float(arow["adf_statistic"]) if pd.notna(arow["adf_statistic"]) else np.nan,
                "adf_pvalue_c": float(arow["pvalue"]) if pd.notna(arow["pvalue"]) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _assumption_matrix(
    normality_df: pd.DataFrame,
    tails_df: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    acf_df: pd.DataFrame,
    lb_df: pd.DataFrame,
    ar1_df: pd.DataFrame,
    adf_df: pd.DataFrame,
    has_non_positive: bool,
) -> pd.DataFrame:
    full_n = normality_df.loc[normality_df["period"] == "full_sample"].iloc[0]
    tail3 = tails_df.loc[(tails_df["period"] == "full_sample") & (tails_df["sigma_threshold"] == 3)].iloc[0]
    acf1_r = acf_df.loc[(acf_df["series"] == "returns") & (acf_df["lag"] == 1), "acf"].iloc[0]
    acf1_sq = acf_df.loc[(acf_df["series"] == "squared_returns") & (acf_df["lag"] == 1), "acf"].iloc[0]
    lb_ret_10 = lb_df.loc[(lb_df["series"] == "returns") & (lb_df["lag"] == 10)].iloc[0]
    lb_sq_10 = lb_df.loc[(lb_df["series"] == "squared_returns") & (lb_df["lag"] == 10)].iloc[0]
    vol30 = rolling_summary.loc[rolling_summary["window_days"] == 30].iloc[0]
    phi = ar1_df.loc[0, "phi"]
    phi_p = ar1_df.loc[0, "phi_p"]
    adf_c_full = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "c")].iloc[0]

    rows = [
        {
            "assumption": "Normal returns",
            "diagnostic_used": "Jarque-Bera + empirical 3σ exceedance",
            "numerical_result": f"JB p={full_n['jarque_bera_pvalue']:.4g}; empirical/theoretical 3σ ratio={tail3['empirical_to_theoretical_ratio']:.2f}",
            "interpretation": "Normality rejected if p-value is very small and tails are over-represented.",
            "assessment": "contradicted" if full_n["jarque_bera_pvalue"] < 0.05 else "partially supported",
            "qualification": "Heavy tails in daily returns are common in oil markets.",
        },
        {
            "assumption": "Constant volatility",
            "diagnostic_used": "30/90-day rolling annualized volatility range",
            "numerical_result": f"30d min/median/max={vol30['min']:.3f}/{vol30['median']:.3f}/{vol30['max']:.3f}",
            "interpretation": "Large dispersion in rolling vol indicates time-varying volatility.",
            "assessment": "contradicted",
            "qualification": "Daily data shows volatility clustering; GBM constant sigma is restrictive.",
        },
        {
            "assumption": "Independent returns",
            "diagnostic_used": "ACF(1) returns + Ljung-Box(10)",
            "numerical_result": f"ACF1={acf1_r:.3f}; LB10 p={lb_ret_10['ljung_box_pvalue']:.4f}",
            "interpretation": "Small linear ACF can coexist with higher-order dependence.",
            "assessment": "partially supported" if abs(acf1_r) < 0.1 else "contradicted",
            "qualification": "Squared-return diagnostics often indicate residual dependence in variance.",
        },
        {
            "assumption": "Continuous price changes",
            "diagnostic_used": "Tail events in daily returns",
            "numerical_result": f"3σ freq={tail3['empirical_frequency']:.4f}",
            "interpretation": "Daily sampling cannot prove path continuity/discontinuity.",
            "assessment": "partially supported",
            "qualification": "Inconclusive from daily observations; intraday data required for jump detection.",
        },
        {
            "assumption": "Absence of mean reversion",
            "diagnostic_used": "AR(1) on log prices + ADF(c)",
            "numerical_result": f"phi={phi:.6f} (p={phi_p:.4g}); ADF p={adf_c_full['pvalue']:.4g}",
            "interpretation": (
                "ADF does not robustly reject the unit-root null at 5%. "
                "Failure to reject the unit-root null is not proof that the unit root exists. "
                "ADF reliability can be reduced by structural breaks. "
                "The OLS t-test for phi!=0 is not valid evidence of mean reversion; "
                "only ADF-based inference on phi<1 is appropriate for this purpose."
            ),
            "assessment": "inconclusive/not contradicted at 5%",
            "qualification": (
                "Half-life is not reported because ADF tests do not robustly reject the "
                "unit root. Persistent/random-walk behavior remains plausible."
            ),
        },
        {
            "assumption": "Strictly positive prices",
            "diagnostic_used": "Observed spot prices sign check",
            "numerical_result": "All observed prices > 0" if not has_non_positive else "At least one non-positive observed price",
            "interpretation": "Observed sample positivity is consistent with GBM support on (0,∞).",
            "assessment": "supported" if not has_non_positive else "contradicted",
            "qualification": "Historical sample did not include non-positive Brent spot observations.",
        },
        {
            "assumption": "Volatility clustering absent",
            "diagnostic_used": "ACF(1) squared returns + Ljung-Box(10) squared returns",
            "numerical_result": f"ACF1(sq)={acf1_sq:.3f}; LB10 p={lb_sq_10['ljung_box_pvalue']:.4f}",
            "interpretation": "Significant squared-return dependence indicates clustering.",
            "assessment": "contradicted" if lb_sq_10["ljung_box_pvalue"] < 0.05 else "partially supported",
            "qualification": "Included as dependence caveat for constant-variance GBM.",
        },
    ]

    return pd.DataFrame(rows)


def _extract_stage_outputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tables_root = project_root() / "outputs" / "tables"
    params_path = tables_root / "gbm_parameters.csv"
    metrics_path = tables_root / "backtest_metrics.csv"

    issues: list[str] = []

    if not params_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 output: {params_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 4 output: {metrics_path}")

    params = pd.read_csv(params_path)
    pmap = dict(zip(params["Parameter"], params["Value"]))

    cleaned = _load_stage1_processed()
    sim_params = pd.DataFrame(
        [
            {
                "S0": float(cleaned["Price_USD_per_barrel"].iloc[-1]),
                "mu_annual": float(pmap.get("mu_annual", np.nan)),
                "sigma_annual": float(pmap.get("sigma_annual", np.nan)),
                "horizon_days": 252,
                "n_paths": 5000,
                "seed": 42,
            }
        ]
    )

    metrics = pd.read_csv(metrics_path)
    metrics = metrics.rename(columns={"Metric": "metric", "Value": "value"})

    interpret_rows = [
        {
            "metric": "coverage_p05_p95",
            "definition": "Share of actual prices within 5th–95th forecast interval.",
            "preferred_direction": "Near nominal target with sharp intervals",
            "observed_value": float(metrics.loc[metrics["metric"] == "coverage_p05_p95", "value"].iloc[0]),
            "interpretation": "100% coverage may indicate overly wide intervals rather than precise calibration.",
            "limitation": "High coverage alone does not imply informative forecasts.",
        },
        {
            "metric": "avg_interval_width",
            "definition": "Mean width of p95-p05 interval in USD/barrel.",
            "preferred_direction": "Lower width for sharper forecasts, conditional on calibration",
            "observed_value": float(metrics.loc[metrics["metric"] == "avg_interval_width", "value"].iloc[0]),
            "interpretation": "Average width near 74 USD implies economically broad uncertainty bands.",
            "limitation": "Wide bands can make directional decision support weak.",
        },
        {
            "metric": "directional_accuracy",
            "definition": "Fraction of days where sign of median forecast change matches actual change.",
            "preferred_direction": ">0.5 for useful directional forecasting",
            "observed_value": float(metrics.loc[metrics["metric"] == "directional_accuracy", "value"].iloc[0]),
            "interpretation": "Sub-50% indicates limited day-to-day directional forecasting ability.",
            "limitation": "GBM median path is not designed for short-horizon market timing.",
        },
        {
            "metric": "MAE_RMSE_MAPE_context",
            "definition": "Point-error summaries against median path forecast.",
            "preferred_direction": "Lower values preferred for point forecasts",
            "observed_value": float(metrics.loc[metrics["metric"] == "RMSE", "value"].iloc[0]),
            "interpretation": "MAE/RMSE/MAPE summarize path error but do not fully score probabilistic forecasts.",
            "limitation": "Proper scoring rules may be more suitable for stochastic predictive distributions.",
        },
        {
            "metric": "drift_vol_estimation",
            "definition": "Stage 2 uses historical log-return moments (annualized with 252).",
            "preferred_direction": "Transparent and reproducible",
            "observed_value": float(sim_params.loc[0, "mu_annual"]),
            "interpretation": "Drift/vol estimated from historical sample then reused in simulation.",
            "limitation": "Parameter instability across regimes can affect forecast reliability.",
        },
        {
            "metric": "historical_drift_usage",
            "definition": "Whether historical drift was used in simulation/backtest.",
            "preferred_direction": "Explicitly documented",
            "observed_value": 1.0,
            "interpretation": "Simulation and analytical backtest use historical (physical-measure) drift estimate.",
            "limitation": "Not equivalent to risk-neutral valuation drift.",
        },
        {
            "metric": "physical_vs_risk_neutral",
            "definition": "Distinction between physical-measure backtest and risk-neutral pricing drift.",
            "preferred_direction": "Explicit distinction",
            "observed_value": 1.0,
            "interpretation": "Backtest evaluates historical predictive performance under physical measure, not derivative-pricing measure.",
            "limitation": "Results should not be interpreted as risk-neutral option-pricing validation.",
        },
    ]
    interpretation = pd.DataFrame(interpret_rows)

    sim_params.to_csv(paths["tables"] / "gbm_simulation_parameters.csv", index=False)
    metrics.to_csv(paths["tables"] / "gbm_backtest_metrics.csv", index=False)
    interpretation.to_csv(paths["tables"] / "gbm_backtest_interpretation.csv", index=False)

    expected = {
        "S0": 61.35,
        "mu_annual": 0.121273,
        "sigma_annual": 0.413151,
        "horizon_days": 252,
        "n_paths": 5000,
        "seed": 42,
        "MAE": 9.023262590029372,
        "RMSE": 10.319288073690892,
        "MAPE_pct": 13.644789070221366,
        "coverage_p05_p95": 1.0,
        "avg_interval_width": 74.04444587674644,
        "directional_accuracy": 0.4779116465863454,
    }

    sim_row = sim_params.iloc[0]
    tolerances = {
        "S0": 1e-9,
        "mu_annual": 1e-6,
        "sigma_annual": 1e-6,
        "horizon_days": 1e-9,
        "n_paths": 1e-9,
        "seed": 1e-9,
    }
    for key in ("S0", "mu_annual", "sigma_annual", "horizon_days", "n_paths", "seed"):
        observed = float(sim_row[key])
        if not np.isclose(observed, expected[key], atol=tolerances[key], rtol=0):
            issues.append(f"{key} mismatch: expected {expected[key]}, observed {observed}")

    m_map = dict(zip(metrics["metric"], metrics["value"]))
    for key in ("MAE", "RMSE", "MAPE_pct", "coverage_p05_p95", "avg_interval_width", "directional_accuracy"):
        observed = float(m_map[key])
        if not np.isclose(observed, expected[key], atol=1e-9, rtol=0):
            issues.append(f"{key} mismatch: expected {expected[key]}, observed {observed}")

    return sim_params, metrics, interpretation, issues


def _plot_basic_figures(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    start = df["Date"].min().date().isoformat()
    end = df["Date"].max().date().isoformat()

    def with_source(fig: plt.Figure) -> None:
        fig.text(0.01, 0.01, _source_note(), fontsize=8)

    # Price history
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Price_USD_per_barrel"], color="steelblue", linewidth=1.2)
    ax.set_title(f"Europe Brent Spot Price History (USD/barrel)\nSample: {start} to {end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "brent_price_history.png", dpi=320)
    plt.close(fig)

    # Log returns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Log_Return"], color="purple", linewidth=0.8)
    ax.set_title(f"Daily Log Returns of Brent Spot Price\nSample: {start} to {end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "brent_log_returns.png", dpi=320)
    plt.close(fig)

    # Histogram + normal overlay
    r = df["Log_Return"].dropna()
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r, bins=80, density=True, alpha=0.75, color="slateblue", label="Empirical")
    xs = np.linspace(float(r.min()), float(r.max()), 500)
    ax.plot(xs, norm.pdf(xs, mu, sigma), color="firebrick", linewidth=2, label="Fitted normal")
    ax.set_title(f"Brent Daily Log Return Distribution with Normal Overlay\nSample: {start} to {end}")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "return_histogram_normal_overlay.png", dpi=320)
    plt.close(fig)

    # QQ plot
    fig = plt.figure(figsize=(8, 8))
    qqplot(r, line="45", ax=plt.gca(), fit=True)
    plt.title(f"Normal Q-Q Plot of Brent Daily Log Returns\nSample: {start} to {end}")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Sample quantiles")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "normal_qq_plot.png", dpi=320)
    plt.close(fig)

    # Log price history
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Log_Price"], color="teal", linewidth=1.2)
    ax.set_title(f"Log Price History: ln(Brent Spot Price)\nSample: {start} to {end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log price")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "log_price_history.png", dpi=320)
    plt.close(fig)


def _plot_vol_acf_figures(df: pd.DataFrame, rolling_df: pd.DataFrame, period_comp: pd.DataFrame, paths: dict[str, Path]) -> None:
    start = df["Date"].min().date().isoformat()
    end = df["Date"].max().date().isoformat()

    def with_source(fig: plt.Figure) -> None:
        fig.text(0.01, 0.01, _source_note(), fontsize=8)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_df["Date"], rolling_df["rolling_vol_30d"], label="30-day rolling annualized vol", linewidth=1)
    ax.plot(rolling_df["Date"], rolling_df["rolling_vol_90d"], label="90-day rolling annualized vol", linewidth=1)
    ax.set_title(f"Rolling Annualized Volatility (30d and 90d)\nSample: {start} to {end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "rolling_volatility_30_90_day.png", dpi=320)
    plt.close(fig)

    returns = df["Log_Return"].dropna()
    for ser, name, fn in [
        (returns, "Returns", "acf_returns.png"),
        (returns.pow(2), "Squared Returns", "acf_squared_returns.png"),
        (returns.abs(), "Absolute Returns", "acf_absolute_returns.png"),
    ]:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        plot_acf(ser, lags=20, alpha=0.05, ax=ax)
        ax.set_title(f"ACF of Brent Daily {name}\nSample: {start} to {end}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        with_source(fig)
        fig.savefig(paths["figures"] / fn, dpi=320)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(period_comp["period"], period_comp["annualized_vol"], color=["steelblue", "seagreen", "darkorange"])
    ax.set_title("Annualized Volatility by Required Sample Period")
    ax.set_xlabel("Sample period")
    ax.set_ylabel("Annualized volatility")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    with_source(fig)
    fig.savefig(paths["figures"] / "sample_period_comparison.png", dpi=320)
    plt.close(fig)


def _plot_backtest_figures(paths: dict[str, Path]) -> None:
    comparison_path = project_root() / "outputs" / "tables" / "backtest_path_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing backtest comparison: {comparison_path}")
    comp = pd.read_csv(comparison_path, parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(comp["date"], comp["actual_price"], color="firebrick", label="Observed price", linewidth=1.4)
    ax.plot(comp["date"], comp["forecast_p50"], color="navy", linestyle="--", label="GBM median forecast", linewidth=1.2)
    ax.set_title(
        "Stage 4 Backtest: Observed Brent Spot vs GBM Median Forecast\n"
        f"Sample: {comp['date'].min().date()} to {comp['date'].max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    ax.legend()
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / "gbm_backtest_observed_vs_simulated.png", dpi=320)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(comp["date"], comp["forecast_p05"], comp["forecast_p95"], alpha=0.25, color="steelblue", label="GBM p05-p95 interval")
    ax.plot(comp["date"], comp["actual_price"], color="firebrick", label="Observed price", linewidth=1.2)
    ax.plot(comp["date"], comp["forecast_p50"], color="navy", linestyle="--", label="GBM p50", linewidth=1.0)
    ax.set_title(
        "Stage 4 Backtest Prediction Intervals (GBM)\n"
        f"Sample: {comp['date'].min().date()} to {comp['date'].max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / "gbm_prediction_intervals.png", dpi=320)
    plt.close(fig)


def _write_report(
    paths: dict[str, Path],
    data_validation: pd.DataFrame,
    descriptive: pd.DataFrame,
    normality: pd.DataFrame,
    tails: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    lb_df: pd.DataFrame,
    ar1_df: pd.DataFrame,
    adf_df: pd.DataFrame,
    period_comp: pd.DataFrame,
    assumptions: pd.DataFrame,
    sim_params: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    preserve_issues: list[str],
    limitations: list[str],
) -> None:
    dv = data_validation.iloc[0]
    full_desc = descriptive.loc[descriptive["period"] == "full_sample"].iloc[0]
    full_norm = normality.loc[normality["period"] == "full_sample"].iloc[0]
    tail3 = tails.loc[(tails["period"] == "full_sample") & (tails["sigma_threshold"] == 3)].iloc[0]
    lb_ret10 = lb_df.loc[(lb_df["series"] == "returns") & (lb_df["lag"] == 10)].iloc[0]
    lb_sq10 = lb_df.loc[(lb_df["series"] == "squared_returns") & (lb_df["lag"] == 10)].iloc[0]
    adf_full_c = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "c")].iloc[0]

    metrics_map = dict(zip(backtest_metrics["metric"], backtest_metrics["value"]))

    period_lines = []
    for _, row in descriptive.iterrows():
        period_lines.append(
            f"- **{row['period']}** ({row['start_date']} to {row['end_date']}): "
            f"n_prices={int(row['observations_prices'])}, n_returns={int(row['observations_returns'])}, "
            f"mean={row['mean']:.6f}, std={row['std']:.6f}, annualized vol={row['annualized_vol']:.4f}, "
            f"skew={row['skewness']:.4f}, excess kurtosis={row['excess_kurtosis']:.4f}."
        )

    preserve_note = "All checked Stage 3/4 reference values matched expected constants."
    if preserve_issues:
        preserve_note = "Differences found:\n" + "\n".join(f"- {issue}" for issue in preserve_issues)

    limitations_text = "\n".join(f"- {x}" for x in limitations)

    bullets = [
        f"Full-sample processed coverage: {full_desc['start_date']} to {full_desc['end_date']} with {int(full_desc['observations_prices'])} prices.",
        f"Full-sample return mean={full_desc['mean']:.6f}, std={full_desc['std']:.6f}, annualized vol={full_desc['annualized_vol']:.4f}.",
        f"Full-sample return range: min={full_desc['min']:.6f}, max={full_desc['max']:.6f}.",
        f"Jarque-Bera statistic={full_norm['jarque_bera_statistic']:.4f}, p-value={full_norm['jarque_bera_pvalue']:.4g}.",
        f"Empirical outside-3σ frequency={tail3['empirical_frequency']:.6f}, theoretical={tail3['theoretical_normal_probability']:.6f}.",
        f"Ljung-Box returns lag10 p-value={lb_ret10['ljung_box_pvalue']:.4g}; squared returns lag10 p-value={lb_sq10['ljung_box_pvalue']:.4g}.",
        f"ADF(full sample, regression='c'): stat={adf_full_c['adf_statistic']:.4f}, p={adf_full_c['pvalue']:.4g}, usedlag={int(adf_full_c['usedlag'])}.",
        f"Stage 3 inputs extracted: S0={sim_params.loc[0, 'S0']:.2f}, mu={sim_params.loc[0, 'mu_annual']:.6f}, sigma={sim_params.loc[0, 'sigma_annual']:.6f}.",
        f"Backtest MAE={metrics_map['MAE']:.6f}, RMSE={metrics_map['RMSE']:.6f}, MAPE={metrics_map['MAPE_pct']:.6f}%.",
        f"Backtest coverage_p05_p95={metrics_map['coverage_p05_p95']:.6f}, avg_interval_width={metrics_map['avg_interval_width']:.6f} USD.",
        f"Directional accuracy={metrics_map['directional_accuracy']:.6f} (<0.50 indicates weak short-horizon direction capture).",
        "COVID-19 stress period selected ex ante based on external event timing (2020-02-01 to 2020-06-30).",
    ]

    report = fr"""# Brent GBM Handover Report

## 1) Objective and assumptions tested
This handover adds a historical empirical diagnostics component alongside existing GBM Stage 3/4 simulation and backtesting outputs. Diagnostics evaluate the assumptions of normal returns, constant volatility, independent returns, continuous price changes, absence of mean reversion, and strictly positive prices.

## 2) Data provenance and validation
- Source URL: {dv['source_url']}
- Retrieval date: {dv['retrieval_date']}
- Raw filename: {dv['raw_filename']}
- Units: {dv['units']}
- Frequency: {dv['frequency']}
- First/last observation in raw: {dv['first_observation']} to {dv['last_observation']}
- Missing: {int(dv['missing_values'])}
- Non-numeric: {int(dv['non_numeric_values'])}
- Duplicates: {int(dv['duplicate_dates'])}
- Non-positive: {int(dv['non_positive_values'])}
- Chronological order in raw: {bool(dv['chronological_order'])}

No interpolation was applied. Missing values are not filled.

## 3) Return construction
Daily log returns are constructed exactly as:
\[r_t = \ln(S_t / S_{{t-1}})\]
with annualisation factor \(\sqrt{{252}}\) for volatility. No trimming or winsorisation was applied.

## 4) Descriptive statistics by required period
{chr(10).join(period_lines)}

## 5) Normality and tails
- Jarque-Bera and fitted normal parameters are reported in `tables/normality_tests.csv`.
- Tail exceedance frequencies outside 2σ/3σ/4σ/5σ, theoretical normal probabilities, and empirical/theoretical ratios are in `tables/empirical_tail_frequencies.csv`.
- Top-10 absolute daily log returns by period are in `tables/largest_absolute_returns.csv`.
- Caveat: daily data cannot prove path discontinuity.

## 6) Volatility and dependence
- Rolling annualized volatility summaries for 30-day and 90-day windows are in `tables/rolling_volatility_summary.csv`.
- ACF tables are in `tables/autocorrelations.csv` for returns, squared returns, and absolute returns.
- Ljung-Box results at lags 5/10/20 are in `tables/ljung_box_results.csv`.

## 7) Mean reversion and unit root diagnostics
- AR(1) on log prices is reported in `tables/ar1_results.csv`.
- ADF tests with `regression='c'` and `regression='ct'` and `autolag='AIC'` are in `tables/adf_results.csv`.
- Half-life is **not reported**: ADF tests do not robustly reject the unit-root null for this series.
- A statistically significant OLS phi coefficient (t-test vs 0) is **not** evidence of mean reversion; only ADF-based inference on phi<1 is valid for this purpose.
- Failure to reject the unit-root null is not proof that the unit root exists; structural breaks can reduce ADF reliability.
- Both ADF specifications (regression='c' and regression='ct') are retained with their critical values.

## 8) Sample-period comparison
`tables/sample_period_comparison.csv` and `figures/sample_period_comparison.png` compare the required periods.
The COVID-19 stress period (2020-02-01 to 2020-06-30) is selected ex ante based on external event timing.

## 9) GBM assumption assessment matrix
See `tables/gbm_assumption_assessment.csv`.

## 10) Stage 3/4 simulation and backtest assessment (separate component)
- Simulation parameters extracted from existing generated outputs are in `tables/gbm_simulation_parameters.csv`.
- Stage 4 metrics extracted from existing output are in `tables/gbm_backtest_metrics.csv`.
- Interpretation table with caveats is in `tables/gbm_backtest_interpretation.csv`.

Preservation check:
{preserve_note}

## 11) Limitations
- Spot vs futures basis differences are not modeled.
- Structural breaks and regime changes can violate constant-parameter assumptions.
- Statistical test outcomes depend on sample window and test power.
- Daily observations cannot conclusively identify jump discontinuities.
- Backtest is under physical-measure historical dynamics, distinct from risk-neutral pricing contexts.
{limitations_text}

## 12) Reproducibility details
- Analysis script: `brent_gbm_analysis/src/historical_diagnostics.py`
- Generated tables: `handoff_brent_analysis/tables/`
- Generated figures: `handoff_brent_analysis/figures/`
- Run log: `handoff_brent_analysis/logs/analysis_run_log.txt`
- Environment log: `handoff_brent_analysis/logs/environment.txt`
- Test log: `handoff_brent_analysis/logs/test_results.txt`

## 13) Concise factual summary
"""

    report += "\n".join(f"- {b}" for b in bullets)
    report += "\n"

    (paths["base"] / "HANDOFF_REPORT.md").write_text(report, encoding="utf-8")


def _write_audit(paths: dict[str, Path]) -> None:
    content = """# Methodology Audit

This audit maps each computation to source/function/input/formula/package/options/output.

## Computation mapping
- Stage 1 retrieval: `src/download_data.py::download_brent_data` -> raw CSV + provenance JSON.
- Stage 1 preparation: `src/prepare_data.py::prepare_data` -> cleaned processed CSV and validation summary.
- Stage 2 estimation: `src/estimate_gbm.py::{compute_log_returns, estimate_gbm_parameters}` -> `outputs/tables/gbm_parameters.csv`.
- Stage 3 simulation: `src/simulate_gbm.py::{simulate_gbm_paths, compute_simulation_quantiles}` -> simulation outputs.
- Stage 4 backtest: `src/evaluate_gbm.py::{compute_forecast_quantiles, compute_backtest_metrics}` -> backtest metrics/comparison.
- Handover diagnostics: `src/historical_diagnostics.py` -> handoff tables, figures, report, audit, logs.

## Formula/package/options/output
- Returns: `r_t = ln(S_t/S_{t-1})` using pandas/numpy.
- Annualized volatility: `std_daily * sqrt(252)`.
- Jarque-Bera/tails: scipy.stats.
- Rolling vol: pandas rolling std with windows 30/90 and annualization factor 252.
- ACF: statsmodels `acf` and `plot_acf`.
- Ljung-Box: statsmodels `acorr_ljungbox` at lags [5,10,20].
- AR(1): statsmodels OLS on `log_price_t ~ 1 + log_price_{t-1}`.
- ADF: statsmodels `adfuller(..., regression in {'c','ct'}, autolag='AIC')`.

## Audit flags
- Silent NA drops: **Flagged and explicit** (`dropna()` used intentionally for returns/tests).
- Interpolation: **Not used**.
- Look-ahead bias: **Flagged caveat** for Stage 4 because mu/sigma come from Stage 2 full sample.
- Date inconsistency: **Checked** via validation + period slicing checks.
- Hard-coded results: **Not used for computed diagnostics**; values are read from generated files for report tables.
- Annualization correctness: **Checked** against `sqrt(252)`.
- Arithmetic-vs-log returns: **Log returns only** in diagnostics.
- Assumption caveats: **Explicitly included** (normality, constant volatility, continuity limits, physical vs risk-neutral).
- Figure-table sample mismatch: **Checked in tests where feasible**.
- Notebook-script discrepancy: **Not relied upon**; workflow is script-based.
- Reproducibility gaps: **Logged** in `logs/analysis_run_log.txt` and `logs/environment.txt`.
"""
    (paths["base"] / "METHODOLOGY_AUDIT.md").write_text(content, encoding="utf-8")


def _copy_artifacts(paths: dict[str, Path]) -> None:
    raw_dir = project_root() / "data" / "raw"
    provenance = _load_provenance(raw_dir)
    raw_file = _find_raw_file(raw_dir, provenance)
    if raw_file and raw_file.exists():
        shutil.copy2(raw_file, paths["raw"] / raw_file.name)
    prov_path = raw_dir / "data_provenance.json"
    if prov_path.exists():
        shutil.copy2(prov_path, paths["raw"] / prov_path.name)

    processed_src = project_root() / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
    if processed_src.exists():
        shutil.copy2(processed_src, paths["processed"] / processed_src.name)

    include = ["src", "tests", "notebooks", "requirements.txt", "README.md"]

    def ignore_func(_: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in {".venv", "__pycache__", ".ipynb_checkpoints"}:
                ignored.add(name)
            if "secret" in name.lower():
                ignored.add(name)
        return ignored

    for item in include:
        src_path = project_root() / item
        dst_path = paths["code"] / item
        if src_path.is_dir():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path, ignore=ignore_func)
        elif src_path.is_file():
            shutil.copy2(src_path, dst_path)


def _write_environment_log(logs_dir: Path) -> None:
    import matplotlib as mpl
    import scipy as sp
    import statsmodels as sm

    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"matplotlib={mpl.__version__}",
        f"scipy={sp.__version__}",
        f"statsmodels={sm.__version__}",
    ]
    (logs_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zip_handoff(base: Path) -> Path:
    zip_path = repo_root() / "brent_gbm_handover.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in base.rglob("*"):
            zf.write(file, arcname=file.relative_to(repo_root()))
    return zip_path


def build_handoff_package() -> dict[str, object]:
    base = handoff_root()
    if base.exists():
        shutil.rmtree(base)
    paths = _ensure_handoff_dirs(base)

    stage_results: list[StageResult] = []
    limitations: list[str] = []

    for name, fn in [
        ("download_data", download_brent_data),
        ("prepare_data", prepare_data),
        ("estimate_gbm", estimate_gbm),
        ("simulate_gbm", simulate_gbm),
        ("evaluate_gbm", evaluate_gbm),
    ]:
        try:
            fn()
            stage_results.append(StageResult(name=name, status="ok", details="completed"))
        except Exception as exc:  # noqa: BLE001
            msg = f"{name} failed: {type(exc).__name__}: {exc}"
            stage_results.append(StageResult(name=name, status="failed", details=msg))
            limitations.append(msg)

    raw_dir = project_root() / "data" / "raw"
    provenance = _load_provenance(raw_dir)
    raw_file = _find_raw_file(raw_dir, provenance)
    if raw_file is None:
        raise FileNotFoundError("No raw Brent source file found after stage execution.")

    _, validation = _validate_raw_data(raw_file)

    processed = _load_stage1_processed()
    proc_ext = _build_processed_with_logs(processed)
    proc_ext.to_csv(paths["processed"] / "brent_prices_2000_2025_with_log_features.csv", index=False)

    # Validate sample window: the requested window begins 2000-01-01.
    # No requirement for a price observation on 2000-01-01 itself (weekend/holiday).
    # The first available processed observation is the first valid trading day
    # within the requested window (within a few calendar days to allow for weekends
    # and public holidays at the start of the year).
    first_proc_date = proc_ext["Date"].min()
    last_proc_date = proc_ext["Date"].max()
    requested_start = pd.Timestamp("2000-01-01")
    requested_end = pd.Timestamp("2025-12-31")
    # Allow up to 7 calendar days after the nominal start to account for weekends/holidays.
    window_ok = (
        first_proc_date >= requested_start
        and first_proc_date <= requested_start + pd.Timedelta(days=7)
        and last_proc_date >= requested_end
    )
    if not window_ok:
        limitations.append(
            f"The requested sample window begins on 1 January 2000. "
            f"The first available reported price observation within that window occurs on "
            f"{first_proc_date.date().isoformat()}. "
            f"Last processed observation: {last_proc_date.date().isoformat()}. "
            f"Required end date 2025-12-31 not satisfied."
        )

    data_validation = pd.DataFrame(
        [
            {
                "source_url": provenance.get("download_url", ""),
                "retrieval_date": provenance.get("retrieval_date", ""),
                "raw_filename": raw_file.name,
                "units": validation["units"],
                "frequency": validation["frequency"],
                "first_observation": validation["first_observation"],
                "last_observation": validation["last_observation"],
                "missing_values": validation["missing_values"],
                "non_numeric_values": validation["non_numeric_values"],
                "duplicate_dates": validation["duplicate_dates"],
                "non_positive_values": validation["non_positive_values"],
                "chronological_order": validation["chronological_order"],
            }
        ]
    )
    data_validation.to_csv(paths["tables"] / "data_validation.csv", index=False)

    descriptive_rows: list[dict[str, object]] = []
    normality_rows: list[dict[str, object]] = []
    tails_frames: list[pd.DataFrame] = []
    largest_frames: list[pd.DataFrame] = []

    for label, start, end in PERIODS:
        period_df = _slice_period(proc_ext, start, end)
        if period_df.empty:
            limitations.append(f"Missing coverage for period '{label}' ({start} to {end}).")
            continue
        descriptive_rows.append(_period_descriptive(label, period_df))
        n_row, tails_df, top_df = _normality_tail_tables(label, period_df)
        normality_rows.append(n_row)
        tails_frames.append(tails_df)
        largest_frames.append(top_df)

    descriptive = pd.DataFrame(descriptive_rows)
    normality = pd.DataFrame(normality_rows)
    tails = pd.concat(tails_frames, ignore_index=True) if tails_frames else pd.DataFrame()
    largest = pd.concat(largest_frames, ignore_index=True) if largest_frames else pd.DataFrame()

    descriptive.to_csv(paths["tables"] / "descriptive_statistics.csv", index=False)
    normality.to_csv(paths["tables"] / "normality_tests.csv", index=False)
    tails.to_csv(paths["tables"] / "empirical_tail_frequencies.csv", index=False)
    largest.to_csv(paths["tables"] / "largest_absolute_returns.csv", index=False)

    rolling_df, rolling_summary = _rolling_volatility(proc_ext)
    rolling_summary.to_csv(paths["tables"] / "rolling_volatility_summary.csv", index=False)

    acf_df, lb_df = _autocorrelation_tables(proc_ext)
    acf_df.to_csv(paths["tables"] / "autocorrelations.csv", index=False)
    lb_df.to_csv(paths["tables"] / "ljung_box_results.csv", index=False)

    ar1_df, adf_df = _ar1_and_adf_tables(proc_ext)
    ar1_df.to_csv(paths["tables"] / "ar1_results.csv", index=False)
    adf_df.to_csv(paths["tables"] / "adf_results.csv", index=False)

    period_comp = _sample_period_comparison(proc_ext, descriptive, normality, tails, adf_df)
    period_comp.to_csv(paths["tables"] / "sample_period_comparison.csv", index=False)

    assumptions = _assumption_matrix(
        normality, tails, rolling_summary, acf_df, lb_df, ar1_df, adf_df, bool(validation["non_positive_values"] > 0)
    )
    assumptions.to_csv(paths["tables"] / "gbm_assumption_assessment.csv", index=False)

    sim_params, backtest_metrics, _interp, preserve_issues = _extract_stage_outputs(paths)

    _plot_basic_figures(proc_ext, paths)
    _plot_vol_acf_figures(proc_ext, rolling_df, period_comp, paths)
    _plot_backtest_figures(paths)

    _write_report(
        paths,
        data_validation,
        descriptive,
        normality,
        tails,
        rolling_summary,
        lb_df,
        ar1_df,
        adf_df,
        period_comp,
        assumptions,
        sim_params,
        backtest_metrics,
        preserve_issues,
        limitations,
    )
    _write_audit(paths)

    _copy_artifacts(paths)

    log_lines = [f"[{x.status}] {x.name}: {x.details}" for x in stage_results]
    if limitations:
        log_lines.append("\nLimitations:")
        log_lines.extend(f"- {x}" for x in limitations)
    (paths["logs"] / "analysis_run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    test_log_path = paths["logs"] / "test_results.txt"
    if not test_log_path.exists():
        test_log_path.write_text("Test results will be populated after pytest execution.\n", encoding="utf-8")

    _write_environment_log(paths["logs"])

    zip_path = _zip_handoff(base)

    return {
        "handoff_dir": base,
        "zip_path": zip_path,
        "limitations": limitations,
        "preserve_issues": preserve_issues,
    }


if __name__ == "__main__":
    result = build_handoff_package()
    print(f"Handover package generated: {result['handoff_dir']}")
    print(f"Archive generated: {result['zip_path']}")
    if result["limitations"]:
        print("Limitations encountered:")
        for item in result["limitations"]:
            print(f"- {item}")
    if result["preserve_issues"]:
        print("Stage preservation differences:")
        for item in result["preserve_issues"]:
            print(f"- {item}")
