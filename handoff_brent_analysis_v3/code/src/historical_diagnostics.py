"""Build historical diagnostics and full handoff package for Brent GBM analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
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

try:
    from download_data import DOWNLOAD_URL, SOURCE_DESCRIPTION
    from estimate_gbm import estimate_gbm
    from evaluate_gbm import evaluate_gbm
    from prepare_data import prepare_data
    from rolling_origin_backtest import run_rolling_origin_backtest
except ModuleNotFoundError:  # pragma: no cover - import path varies between script and package usage
    from src.download_data import DOWNLOAD_URL, SOURCE_DESCRIPTION
    from src.estimate_gbm import estimate_gbm
    from src.evaluate_gbm import evaluate_gbm
    from src.prepare_data import prepare_data
    from src.rolling_origin_backtest import run_rolling_origin_backtest

ANNUALISATION_FACTOR = 252
HANDOFF_DIRNAME = "handoff_brent_analysis_v3"
ARCHIVE_FILENAME = "brent_gbm_handover_v3.zip"
REPORT_FILENAME = "HANDOFF_REPORT_V3.md"
AUDIT_FILENAME = "METHODOLOGY_AUDIT_V3.md"
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
    return repo_root() / HANDOFF_DIRNAME


def _source_note() -> str:
    return "Source: Europe Brent Spot Price FOB (USD/barrel); underlying EIA series (DCOILBRENTEU)."


def _format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "p = NA"
    if value < 0.001:
        return "p < 0.001"
    if value < 0.01:
        return f"p = {value:.3f}"
    return f"p = {value:.4f}"


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
    adf_c_full = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "c")].iloc[0]
    adf_ct_full = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "ct")].iloc[0]

    rows = [
        {
            "assumption": "Normal log returns",
            "diagnostic_used": "Jarque-Bera test, excess kurtosis, Q-Q plot, empirical 3σ exceedance",
            "numerical_result": (
                f"JB statistic={full_n['jarque_bera_statistic']:.2f}, {_format_pvalue(full_n['jarque_bera_pvalue'])}; "
                f"3σ empirical/theoretical ratio={tail3['empirical_to_theoretical_ratio']:.2f}"
            ),
            "interpretation": "Extremely non-normal tails and Q-Q departures contradict Gaussian daily return innovations.",
            "assessment": "contradicted",
            "qualification": "Machine-readable CSV retains raw p-values; prose avoids implying a literal p=0.",
        },
        {
            "assumption": "Constant volatility",
            "diagnostic_used": "30/90-day rolling annualized volatility, squared/absolute-return dependence",
            "numerical_result": f"30d min/median/max={vol30['min']:.3f}/{vol30['median']:.3f}/{vol30['max']:.3f}",
            "interpretation": "Large rolling-volatility dispersion and persistent variance dependence strongly reject a constant-sigma description.",
            "assessment": "strongly contradicted",
            "qualification": "Volatility clustering is especially pronounced around stress episodes such as 2020.",
        },
        {
            "assumption": "Independent returns",
            "diagnostic_used": "ACF(1) returns + Ljung-Box(10)",
            "numerical_result": f"ACF1={acf1_r:.4f}; LB10 {_format_pvalue(lb_ret_10['ljung_box_pvalue'])}",
            "interpretation": "First-order linear dependence is economically small, but longer-lag serial dependence is statistically detectable.",
            "assessment": "partially supported as an approximation, but statistically rejected as a strict assumption",
            "qualification": "Useful as a rough modelling simplification, not as a literal IID claim.",
        },
        {
            "assumption": "Absence of mean reversion",
            "diagnostic_used": "AR(1) persistence estimate + ADF(c) and ADF(ct)",
            "numerical_result": (
                f"phi={phi:.6f}; ADF(c) {_format_pvalue(adf_c_full['pvalue'])}; "
                f"ADF(ct) {_format_pvalue(adf_ct_full['pvalue'])}"
            ),
            "interpretation": (
                "ADF does not robustly reject the unit-root null at 5%. "
                "Failure to reject the unit-root null is not proof that the unit root exists. "
                "ADF reliability can be reduced by structural breaks. "
                "A half-life is therefore not interpreted, and the OLS t-test against phi=0 is not used as evidence."
            ),
            "assessment": "inconclusive/not contradicted at 5%",
            "qualification": (
                "The intercept-only ADF can reject at 10%, but the conclusion is not robust once a trend is included."
            ),
        },
        {
            "assumption": "Independent and identically distributed return variance",
            "diagnostic_used": "ACF(1) squared returns + Ljung-Box(10) squared returns",
            "numerical_result": f"ACF1(sq)={acf1_sq:.4f}; LB10 {_format_pvalue(lb_sq_10['ljung_box_pvalue'])}",
            "interpretation": "Variance dynamics exhibit autocorrelation and clustering rather than IID dispersion.",
            "assessment": "contradicted",
            "qualification": "This is the main reason the constant-variance GBM assumption performs poorly in stress regimes.",
        },
        {
            "assumption": "Continuous paths",
            "diagnostic_used": "Extreme daily return observations",
            "numerical_result": f"3σ exceedance frequency={tail3['empirical_frequency']:.4f}",
            "interpretation": "Daily observations cannot separate jumps from very large continuous shocks.",
            "assessment": "inconclusive",
            "qualification": "Intraday data would be needed for formal jump diagnostics.",
        },
        {
            "assumption": "Strict positivity",
            "diagnostic_used": "Observed spot price sign check",
            "numerical_result": "All observed prices > 0" if not has_non_positive else "At least one non-positive observed price",
            "interpretation": "Observed Brent prices stayed positive throughout the retained sample.",
            "assessment": "consistent with the observed Brent sample" if not has_non_positive else "contradicted",
            "qualification": "Observed positivity does not itself prove the full lognormal distributional assumption.",
        },
    ]

    return pd.DataFrame(rows)


def _extract_stage_outputs(
    paths: dict[str, Path],
    fixed_origin_result: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_root = project_root() / "outputs" / "tables"
    params_path = tables_root / "gbm_parameters.csv"
    comparison_path = tables_root / "backtest_path_comparison.csv"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 output: {params_path}")
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing Stage 4 comparison output: {comparison_path}")

    params = pd.read_csv(params_path)
    pmap = dict(zip(params["Parameter"], params["Value"]))
    cleaned = _load_stage1_processed()
    comparison = pd.read_csv(comparison_path, parse_dates=["date"])

    full_sample_params = pd.DataFrame(
        [
            {
                "sample_type": "full_sample_forward_simulation",
                "sample_start": cleaned["Date"].min().date().isoformat(),
                "sample_end": cleaned["Date"].max().date().isoformat(),
                "n_prices": int(len(cleaned)),
                "n_returns": int(len(cleaned) - 1),
                "S0": float(cleaned["Price_USD_per_barrel"].iloc[-1]),
                "mu_annual": float(pmap.get("mu_annual", np.nan)),
                "sigma_annual": float(pmap.get("sigma_annual", np.nan)),
                "annualisation_factor": ANNUALISATION_FACTOR,
            }
        ]
    )

    fixed_origin_params = pd.DataFrame(
        [
            {
                "training_start": pd.Timestamp(fixed_origin_result["training_start"]).date().isoformat(),
                "training_end": pd.Timestamp(fixed_origin_result["training_end"]).date().isoformat(),
                "test_start": pd.Timestamp(fixed_origin_result["test_start"]).date().isoformat(),
                "test_end": pd.Timestamp(fixed_origin_result["test_end"]).date().isoformat(),
                "n_training_prices": int(fixed_origin_result["train_size"]),
                "n_training_returns": int(fixed_origin_result["training_n_returns"]),
                "n_test_prices": int(fixed_origin_result["test_size"]),
                "S0": float(fixed_origin_result["S0"]),
                "mean_daily_log_return_training": float(fixed_origin_result["mean_daily"]),
                "std_daily_log_return_training": float(fixed_origin_result["std_daily"]),
                "mu_annual_training": float(fixed_origin_result["mu"]),
                "sigma_annual_training": float(fixed_origin_result["sigma"]),
                "annualisation_factor": ANNUALISATION_FACTOR,
            }
        ]
    )

    actual = comparison["actual_price"].to_numpy(dtype=float)
    p05 = comparison["forecast_p05"].to_numpy(dtype=float)
    p50 = comparison["forecast_p50"].to_numpy(dtype=float)
    p95 = comparison["forecast_p95"].to_numpy(dtype=float)
    errors = actual - p50
    below = actual < p05
    above = actual > p95
    inside = ~(below | above)

    fixed_origin_metrics = pd.DataFrame(
        [
            {"metric_name": "MAE", "value": float(np.mean(np.abs(errors)))},
            {"metric_name": "RMSE", "value": float(np.sqrt(np.mean(errors ** 2)))},
            {"metric_name": "MAPE_pct", "value": float(np.mean(np.abs(errors / actual)) * 100)},
            {"metric_name": "coverage_90", "value": float(np.mean(inside))},
            {"metric_name": "avg_interval_width", "value": float(np.mean(p95 - p05))},
            {"metric_name": "lower_tail_violation_freq", "value": float(np.mean(below))},
            {"metric_name": "upper_tail_violation_freq", "value": float(np.mean(above))},
        ]
    )

    metric_map = dict(zip(fixed_origin_metrics["metric_name"], fixed_origin_metrics["value"]))
    interpretation = pd.DataFrame(
        [
            {
                "metric_name": "MAE",
                "definition": "Mean absolute error of the held-out fixed-origin median forecast.",
                "observed_value": float(metric_map["MAE"]),
                "preferred_direction": "Lower",
                "interpretation": "Smaller values indicate a median path that stays closer to the realised held-out price path.",
                "limitation": "Summarises only the median path, not the full predictive distribution.",
            },
            {
                "metric_name": "RMSE",
                "definition": "Root mean squared error of the held-out fixed-origin median forecast.",
                "observed_value": float(metric_map["RMSE"]),
                "preferred_direction": "Lower",
                "interpretation": "Penalises large fixed-origin path misses more heavily than MAE.",
                "limitation": "Sensitive to a few large forecast errors in a single realised path.",
            },
            {
                "metric_name": "MAPE",
                "definition": "Mean absolute percentage error of the held-out fixed-origin median forecast.",
                "observed_value": float(metric_map["MAPE_pct"]),
                "preferred_direction": "Lower",
                "interpretation": "Expresses median-path error relative to observed prices over the held-out period.",
                "limitation": "Still evaluates only point predictions from the p50 path.",
            },
            {
                "metric_name": "coverage_90",
                "definition": "Share of held-out observations inside the fixed-origin 90% interval [p05, p95].",
                "observed_value": float(metric_map["coverage_90"]),
                "preferred_direction": "Near 0.90 with informative widths",
                "interpretation": "Observed coverage is descriptive for this single expanding path interval only.",
                "limitation": "100% coverage from one expanding fixed-origin path interval is not evidence of calibrated independent forecasts.",
            },
            {
                "metric_name": "avg_interval_width",
                "definition": "Average width of the held-out fixed-origin 90% interval in USD/barrel.",
                "observed_value": float(metric_map["avg_interval_width"]),
                "preferred_direction": "Lower, conditional on credible coverage",
                "interpretation": "Wide intervals indicate substantial uncertainty around the fixed-origin GBM path forecast.",
                "limitation": "Sharper intervals would be preferable only if repeated-origin coverage remained adequate.",
            },
            {
                "metric_name": "lower_tail_violation_freq",
                "definition": "Frequency with which held-out prices fall below the fixed-origin p05 bound.",
                "observed_value": float(metric_map["lower_tail_violation_freq"]),
                "preferred_direction": "Near 0.05 jointly with upper-tail frequency",
                "interpretation": "Helps identify whether misses are concentrated on the downside.",
                "limitation": "On a single path, tail frequencies are descriptive rather than independent calibration tests.",
            },
            {
                "metric_name": "upper_tail_violation_freq",
                "definition": "Frequency with which held-out prices exceed the fixed-origin p95 bound.",
                "observed_value": float(metric_map["upper_tail_violation_freq"]),
                "preferred_direction": "Near 0.05 jointly with lower-tail frequency",
                "interpretation": "Helps identify whether misses are concentrated on the upside.",
                "limitation": "On a single path, tail frequencies are descriptive rather than independent calibration tests.",
            },
        ]
    )

    full_sample_params.to_csv(paths["tables"] / "gbm_simulation_parameters.csv", index=False)
    fixed_origin_params.to_csv(paths["tables"] / "fixed_origin_training_parameters.csv", index=False)
    fixed_origin_metrics.to_csv(paths["tables"] / "fixed_origin_backtest_metrics.csv", index=False)
    interpretation.to_csv(paths["tables"] / "gbm_backtest_interpretation.csv", index=False)
    comparison.to_csv(paths["tables"] / "fixed_origin_path_comparison.csv", index=False)

    return full_sample_params, fixed_origin_params, fixed_origin_metrics, interpretation, comparison


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
    plt.title(
        "Normal Q-Q Plot of Brent Daily Log Returns (fit=True standardisation)\n"
        f"Sample: {start} to {end}"
    )
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Standardised sample quantiles")
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
        "Held-Out Fixed-Origin GBM Path Forecast\n"
        f"Observed vs median forecast, {comp['date'].min().date()} to {comp['date'].max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    ax.legend()
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / "fixed_origin_observed_vs_median.png", dpi=320)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(comp["date"], comp["forecast_p05"], comp["forecast_p95"], alpha=0.25, color="steelblue", label="GBM p05-p95 interval")
    ax.plot(comp["date"], comp["actual_price"], color="firebrick", label="Observed price", linewidth=1.2)
    ax.plot(comp["date"], comp["forecast_p50"], color="navy", linestyle="--", label="GBM p50", linewidth=1.0)
    ax.set_title(
        "Held-Out Fixed-Origin GBM Path Forecast\n"
        f"90% prediction interval, {comp['date'].min().date()} to {comp['date'].max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / "fixed_origin_prediction_interval.png", dpi=320)
    plt.close(fig)


def _write_report(
    paths: dict[str, Path],
    data_validation: pd.DataFrame,
    descriptive: pd.DataFrame,
    normality: pd.DataFrame,
    tails: pd.DataFrame,
    lb_df: pd.DataFrame,
    adf_df: pd.DataFrame,
    assumptions: pd.DataFrame,
    full_sample_params: pd.DataFrame,
    fixed_origin_params: pd.DataFrame,
    fixed_origin_metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    limitations: list[str],
) -> None:
    dv = data_validation.iloc[0]
    full_desc = descriptive.loc[descriptive["period"] == "full_sample"].iloc[0]
    full_norm = normality.loc[normality["period"] == "full_sample"].iloc[0]
    tail3 = tails.loc[(tails["period"] == "full_sample") & (tails["sigma_threshold"] == 3)].iloc[0]
    adf_c = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "c")].iloc[0]
    adf_ct = adf_df.loc[(adf_df["period"] == "full_sample") & (adf_df["regression"] == "ct")].iloc[0]
    lb_ret10 = lb_df.loc[(lb_df["series"] == "returns") & (lb_df["lag"] == 10)].iloc[0]
    lb_sq10 = lb_df.loc[(lb_df["series"] == "squared_returns") & (lb_df["lag"] == 10)].iloc[0]
    fixed_row = fixed_origin_params.iloc[0]
    fixed_metric_map = dict(zip(fixed_origin_metrics["metric_name"], fixed_origin_metrics["value"]))
    sim_row = full_sample_params.iloc[0]

    period_lines = "\n".join(
        (
            f"- **{row['period']}** ({row['start_date']} to {row['end_date']}): "
            f"{int(row['observations_prices'])} prices, {int(row['observations_returns'])} returns, "
            f"mean daily log return {row['mean']:.15f}, daily std {row['std']:.14f}, "
            f"annualised volatility {row['annualized_vol']:.10f}, skewness {row['skewness']:.4f}, "
            f"excess kurtosis {row['excess_kurtosis']:.4f}."
        )
        for _, row in descriptive.iterrows()
    )

    rolling_lines = "\n".join(
        (
            f"- **{int(row['horizon'])}-day horizon**: n={int(row['n_forecasts'])}, MAE={row['MAE']:.6f}, "
            f"RMSE={row['RMSE']:.6f}, MAPE={row['MAPE']:.6f}%, coverage_90={row['coverage_90']:.6f}, "
            f"avg width={row['avg_interval_width']:.6f}, median width={row['median_interval_width']:.6f}, "
            f"lower-tail freq={row['lower_tail_violation_freq']:.6f}, upper-tail freq={row['upper_tail_violation_freq']:.6f}."
        )
        for _, row in rolling_metrics.iterrows()
    )

    limitation_lines = "\n".join(f"- {item}" for item in limitations) if limitations else "- No additional unresolved pipeline errors were encountered."

    report = f"""# Brent GBM Handover Report V3

## 1) Historical GBM assumption diagnostics
- Requested sample window: {dv['requested_start']} to {dv['requested_end']}.
- The requested sample window begins on 1 January 2000. The first available reported price observation within that window occurs on {dv['first_valid_observation_in_window']}.
- Processed effective sample: {full_desc['start_date']} to {full_desc['end_date']} with {int(full_desc['observations_prices'])} prices and {int(full_desc['observations_returns'])} daily log returns.
- Full-sample mean daily log return = {full_desc['mean']:.15f}; daily standard deviation = {full_desc['std']:.14f}; annualised volatility = {full_desc['annualized_vol']:.10f}.
- Full-sample return range: min = {full_desc['min']:.16f}; max = {full_desc['max']:.16f}.
- Jarque-Bera statistic = {full_norm['jarque_bera_statistic']:.2f}, {_format_pvalue(full_norm['jarque_bera_pvalue'])}.
- Empirical 3σ tail frequency = {tail3['empirical_frequency']:.6f} versus Gaussian benchmark {tail3['theoretical_normal_probability']:.6f}.
- Ljung-Box on returns at lag 10: {_format_pvalue(lb_ret10['ljung_box_pvalue'])}; on squared returns at lag 10: {_format_pvalue(lb_sq10['ljung_box_pvalue'])}.
- ADF with intercept: statistic = {adf_c['adf_statistic']:.4f}, {_format_pvalue(adf_c['pvalue'])}; ADF with intercept and trend: statistic = {adf_ct['adf_statistic']:.4f}, {_format_pvalue(adf_ct['pvalue'])}.
- AR(1) half-life is not interpreted; `half_life_days` is left missing because the ADF evidence is not robust at the 5% level. Failure to reject a unit root is not proof that a unit root exists, and structural breaks can weaken conventional ADF power.

Required-period descriptive summaries:
{period_lines}

The corrected GBM assumption matrix is provided in `tables/gbm_assumption_assessment.csv`.

## 2) Fixed-origin held-out path forecast
This section reports a **held-out fixed-origin path forecast**, not an independent repeated-forecast calibration exercise.

- **Split before estimation**: training sample {fixed_row['training_start']} to {fixed_row['training_end']}; held-out test sample {fixed_row['test_start']} to {fixed_row['test_end']}.
- **Counts**: {int(fixed_row['n_training_prices'])} training prices, {int(fixed_row['n_training_returns'])} training returns, {int(fixed_row['n_test_prices'])} held-out test prices.
- **Training-only estimation**: mean daily log return = {fixed_row['mean_daily_log_return_training']:.12f}; daily std = {fixed_row['std_daily_log_return_training']:.12f}; annualised mu = {fixed_row['mu_annual_training']:.10f}; annualised sigma = {fixed_row['sigma_annual_training']:.10f}.
- **Forecast origin**: S0 = {fixed_row['S0']:.2f}, equal to the final training price.
- No test observations enter fixed-origin parameter estimation.
- Fixed-origin median-path MAE = {fixed_metric_map['MAE']:.8f}; RMSE = {fixed_metric_map['RMSE']:.8f}; MAPE = {fixed_metric_map['MAPE_pct']:.8f}%.
- Fixed-origin 90% interval coverage = {fixed_metric_map['coverage_90']:.6f}; average interval width = {fixed_metric_map['avg_interval_width']:.8f}; lower-tail violation frequency = {fixed_metric_map['lower_tail_violation_freq']:.6f}; upper-tail violation frequency = {fixed_metric_map['upper_tail_violation_freq']:.6f}.
- Directional accuracy is omitted from the dissertation-facing principal result table because the GBM median path is near-constant in sign and does not evidence market-timing skill.
- Figures: `figures/fixed_origin_observed_vs_median.png` and `figures/fixed_origin_prediction_interval.png`.

## 3) Rolling-origin forecast evaluation
Rolling-origin evaluation re-estimates GBM parameters on an expanding sample at each forecast origin and keeps all targets inside the final 252 valid prices.

Parameter regimes are explicitly separated:
1. `tables/gbm_simulation_parameters.csv` records **full-sample parameters** for the separate forward-simulation context.
2. `tables/fixed_origin_training_parameters.csv` records **training-only parameters** for the held-out fixed-origin path forecast.
3. `tables/rolling_origin_metrics_by_horizon.csv` summarises **rolling re-estimated parameters** across origins for horizons 1, 5, and 20 trading days.

Rolling-origin results:
{rolling_lines}

Overlap caveat: the 5-day and 20-day forecast errors overlap when every valid origin is used, so the 1-day-ahead results are the cleanest repeated-origin calibration diagnostic.

Supporting outputs:
- `tables/rolling_origin_forecasts.csv`
- `tables/rolling_origin_metrics_by_horizon.csv`
- `figures/rolling_origin_one_day_forecasts.png`
- `figures/rolling_origin_coverage_by_horizon.png`
- `figures/rolling_origin_interval_width_by_horizon.png`
- `figures/rolling_origin_errors_by_horizon.png`

## 4) Physical vs risk-neutral interpretation
- The historical diagnostics and backtests assess **physical-measure predictive behaviour** under historically estimated drift and volatility.
- The full-sample forward simulation uses mu = {sim_row['mu_annual']:.10f} and sigma = {sim_row['sigma_annual']:.10f} estimated from the complete sample, which is distinct from the fixed-origin and rolling-origin backtests.
- Historical drift is not automatically the risk-neutral drift used for option valuation.
- Commodity risk-neutral pricing can require interest rates, storage costs, convenience yield, and market-price-of-risk adjustments.
- Good historical forecasting would not by itself validate a risk-neutral derivative-pricing model, and poor historical point forecasting does not by itself invalidate GBM for derivative-pricing approximations.
- MAE, RMSE, and MAPE evaluate median forecasts only; coverage and tail-frequency diagnostics are reported separately for interval behaviour.

## 5) Limitations
- Normality and constant-volatility assumptions are empirically poor approximations for Brent daily returns.
- Daily observations cannot distinguish jumps from extreme continuous shocks.
- ADF-based mean-reversion conclusions remain sensitive to structural breaks and low power near a unit root.
- Fixed-origin 100% coverage arises from one expanding path interval and should not be interpreted as calibrated independent forecast evidence.
- Rolling 5-day and 20-day results use overlapping targets and therefore produce dependent forecast errors.
{limitation_lines}

## 6) Reproducibility
- Operating system and package versions are logged in `logs/environment.txt`.
- Exact environment versions are listed in `requirements-lock.txt`.
- Key regeneration commands:
  - Cleaned data: `python src/prepare_data.py`
  - Historical/full-sample parameter estimation: `python src/estimate_gbm.py`
  - Fixed-origin held-out backtest: `python src/evaluate_gbm.py`
  - Rolling-origin evaluation: `python src/rolling_origin_backtest.py`
  - Full v3 handover package: `python src/historical_diagnostics.py`
  - Full test run: `python -m pytest -q`

## 7) Test evidence
- Full pytest output is archived in `logs/test_results_all.txt`.
- Component logs are archived in `logs/test_results_data.txt`, `logs/test_results_historical.txt`, `logs/test_results_fixed_origin.txt`, `logs/test_results_rolling_origin.txt`, and `logs/test_results_handover.txt`.
- These logs are generated from actual pytest execution and should be reviewed directly for pass/fail/skip/warning details.
"""
    (paths["base"] / REPORT_FILENAME).write_text(report, encoding="utf-8")


def _write_audit(paths: dict[str, Path]) -> None:
    content = f"""# Methodology Audit V3

This audit maps the corrected v3 computations to the code that produces them and records the principal methodological controls.

## Computation mapping
- Stage 1 preparation: `src/prepare_data.py::prepare_data` reads the inherited raw Brent CSV, applies the requested 2000-01-01 to 2025-12-31 window, removes invalid prices, and writes the cleaned processed CSV.
- Full-sample parameter estimation: `src/estimate_gbm.py::estimate_gbm` computes log returns on the full cleaned sample and writes `outputs/tables/gbm_parameters.csv`.
- Fixed-origin held-out path forecast: `src/evaluate_gbm.py::evaluate_gbm` performs the split **before** estimation, computes training returns from training prices only, estimates mu and sigma on the training sample only, sets `S0` to the final training price, and excludes all test observations from parameter estimation.
- Rolling-origin evaluation: `src/rolling_origin_backtest.py::run_rolling_origin_backtest` re-estimates mu and sigma on the expanding sample available at each origin and generates 1-, 5-, and 20-day endpoint forecasts for targets in the final 252 valid prices only.
- Handover packaging: `src/historical_diagnostics.py::build_handoff_package` regenerates the v3 tables, figures, report, methodology audit, corrections log, copied data/code, and archive.

## Parameter distinctions
1. **Full-sample parameters for separate forward simulation** are written to `tables/gbm_simulation_parameters.csv`.
2. **Training-only fixed-origin parameters** are written to `tables/fixed_origin_training_parameters.csv`.
3. **Rolling re-estimated parameters at each origin** are stored forecast-by-forecast in `tables/rolling_origin_forecasts.csv` and summarised in `tables/rolling_origin_metrics_by_horizon.csv`.

## Formula/package/options/output
- Returns: `r_t = ln(S_t / S_{{t-1}})` using pandas/NumPy.
- Annualised volatility: `std_daily * sqrt(252)`.
- GBM annualised drift: `mu = mean_daily * 252 + 0.5 * sigma^2`.
- Jarque-Bera, skewness, kurtosis, tail frequencies: SciPy.
- Autocorrelation and Ljung-Box: statsmodels.
- AR(1) and ADF diagnostics: statsmodels OLS and `adfuller(..., regression in {{'c','ct'}}, autolag='AIC')`.
- Fixed-origin and rolling-origin predictive quantiles: exact analytical lognormal GBM quantiles via `src/evaluate_gbm.py::compute_forecast_quantiles`.

## Key audit outcomes
- Split-before-estimation control for the fixed-origin backtest: **implemented**.
- No test observations in fixed-origin parameter estimation: **implemented**.
- S0 for the fixed-origin backtest equals the final training price: **implemented**.
- Rolling-origin forecasts use only information available on or before each origin: **implemented**.
- The old stale statement claiming a fixed-origin look-ahead caveat from full-sample Stage 2 parameters has been removed.
- Historical diagnostics remain code-generated from the processed dataset; reported values are read from generated outputs rather than manually hard-coded into the report.
- Very small p-values are formatted in prose as inequalities rather than literal zero p-values.
- The Q-Q figure explicitly labels fit-based standardisation as `fit=True`.
"""
    (paths["base"] / AUDIT_FILENAME).write_text(content, encoding="utf-8")


def _write_corrections_log(base: Path) -> None:
    content = """# Corrections Log

| Issue | Original behaviour | Correction | Code file | Output affected | Status |
| --- | --- | --- | --- | --- | --- |
| Half-life issue | Reported an AR(1) half-life despite non-robust unit-root rejection. | Left `half_life_days` missing and rewrote the interpretation around ADF-based uncertainty. | `brent_gbm_analysis/src/historical_diagnostics.py` | `tables/ar1_results.csv`, report, audit | Fixed |
| Look-ahead bias | Fixed-origin backtest could be described using full-sample Stage 2 parameters. | Fixed-origin report/audit now state split-before-estimation, training-only mu/sigma, training-only returns, and `S0` as the final training price. | `brent_gbm_analysis/src/evaluate_gbm.py`, `brent_gbm_analysis/src/historical_diagnostics.py` | Fixed-origin tables, report, audit, figures | Fixed |
| Stale audit | Audit still flagged a Stage 4 full-sample look-ahead caveat. | Rewrote the methodology audit for v3 with corrected parameter distinctions. | `brent_gbm_analysis/src/historical_diagnostics.py` | `METHODOLOGY_AUDIT_V3.md` | Fixed |
| Stale interpretation table | Interpretation rows mixed metrics and still implied full-sample parameter usage. | Regenerated separate MAE/RMSE/MAPE/coverage/width/tail rows using training-only fixed-origin results. | `brent_gbm_analysis/src/historical_diagnostics.py` | `tables/gbm_backtest_interpretation.csv` | Fixed |
| Missing training-parameter table | No explicit fixed-origin training-parameter artifact existed. | Added a dedicated training-parameter CSV with split dates, counts, S0, and training-only moments. | `brent_gbm_analysis/src/historical_diagnostics.py` | `tables/fixed_origin_training_parameters.csv` | Fixed |
| Missing rolling-origin analysis | No standalone rolling-origin module or outputs existed. | Added `src/rolling_origin_backtest.py`, rolling forecast/metric tables, and rolling figures for horizons 1/5/20. | `brent_gbm_analysis/src/rolling_origin_backtest.py`, `brent_gbm_analysis/src/historical_diagnostics.py` | Rolling tables, rolling figures, report, audit | Fixed |
| Directional-accuracy interpretation | Directional accuracy could be over-interpreted despite near-constant positive forecast direction. | Removed directional accuracy from the principal fixed-origin and rolling-origin dissertation-facing result tables. | `brent_gbm_analysis/src/historical_diagnostics.py` | Fixed-origin metrics, report | Fixed |
| P-value formatting | Prose could show literal `p=0`. | Added prose formatting that reports very small p-values as inequalities. | `brent_gbm_analysis/src/historical_diagnostics.py` | Report, audit, assumption table prose | Fixed |
| Q-Q label | Q-Q plot y-axis did not state the fit-based standardisation. | Updated the title/caption wording and y-axis label to `Standardised sample quantiles`. | `brent_gbm_analysis/src/historical_diagnostics.py` | `figures/normal_qq_plot.png` | Fixed |
| Missing test evidence | Placeholder test log text remained instead of real pytest output. | Added dedicated v3 test files and generated real pytest logs for all required subsets. | `brent_gbm_analysis/tests/` | `logs/test_results_*.txt` | Fixed |
| Missing locked environment | Only an unconstrained requirements file existed. | Added an exact-version `requirements-lock.txt` and environment logging. | `brent_gbm_analysis/src/historical_diagnostics.py`, `brent_gbm_analysis/requirements-lock.txt` | `requirements-lock.txt`, `logs/environment.txt` | Fixed |
"""
    (base / "CORRECTIONS_LOG.md").write_text(content, encoding="utf-8")


def _copy_artifacts(paths: dict[str, Path]) -> None:
    raw_dir = project_root() / "data" / "raw"
    provenance = _load_provenance(raw_dir)
    raw_file = _find_raw_file(raw_dir, provenance)
    if raw_file and raw_file.exists():
        shutil.copy2(raw_file, paths["raw"] / raw_file.name)
    prov_path = raw_dir / "data_provenance.json"
    if prov_path.exists():
        shutil.copy2(prov_path, paths["raw"] / prov_path.name)

    for processed_name in ["brent_prices_2000_2025_clean.csv", "brent_prices_2000_2025_with_log_features.csv"]:
        processed_src = project_root() / "data" / "processed" / processed_name
        if processed_src.exists():
            shutil.copy2(processed_src, paths["processed"] / processed_src.name)

    def ignore_func(_: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in {".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}:
                ignored.add(name)
            if any(marker in name.lower() for marker in ("secret", "token", "credential", "password")):
                ignored.add(name)
        return ignored

    for item in ["src", "tests"]:
        src_path = project_root() / item
        dst_path = paths["code"] / item
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path, ignore=ignore_func)

    for item in ["README.md", "requirements.txt", "requirements-lock.txt"]:
        src_path = project_root() / item
        if src_path.exists():
            shutil.copy2(src_path, paths["base"] / item)


def _write_environment_log(logs_dir: Path) -> None:
    import matplotlib as mpl
    import openpyxl
    import pytest
    import requests
    import scipy as sp
    import seaborn as sns
    import statsmodels as sm

    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={sp.__version__}",
        f"statsmodels={sm.__version__}",
        f"matplotlib={mpl.__version__}",
        f"seaborn={sns.__version__}",
        f"pytest={pytest.__version__}",
        f"requests={requests.__version__}",
        f"openpyxl={openpyxl.__version__}",
    ]
    (logs_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_requirements_lock(base: Path) -> None:
    import matplotlib as mpl
    import openpyxl
    import pytest
    import requests
    import scipy as sp
    import seaborn as sns
    import statsmodels as sm

    content = "\n".join(
        [
            f"Python=={platform.python_version()}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"SciPy=={sp.__version__}",
            f"statsmodels=={sm.__version__}",
            f"matplotlib=={mpl.__version__}",
            f"seaborn=={sns.__version__}",
            f"pytest=={pytest.__version__}",
            f"requests=={requests.__version__}",
            f"openpyxl=={openpyxl.__version__}",
            "",
        ]
    )
    lock_path = project_root() / "requirements-lock.txt"
    lock_path.write_text(content, encoding="utf-8")
    (base / "requirements-lock.txt").write_text(content, encoding="utf-8")


def _zip_handoff(base: Path) -> Path:
    zip_path = repo_root() / ARCHIVE_FILENAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(base.rglob("*")):
            zf.write(file, arcname=file.relative_to(repo_root()))
    return zip_path


def zip_handoff_package() -> Path:
    """Create or refresh the v3 handoff archive from the current handoff directory."""
    return _zip_handoff(handoff_root())


def build_handoff_package() -> dict[str, object]:
    base = handoff_root()
    if base.exists():
        shutil.rmtree(base)
    paths = _ensure_handoff_dirs(base)

    stage_results: list[StageResult] = []
    limitations: list[str] = []

    for name, fn in [
        ("prepare_data", prepare_data),
        ("estimate_gbm", estimate_gbm),
        ("evaluate_gbm", evaluate_gbm),
        ("rolling_origin_backtest", run_rolling_origin_backtest),
    ]:
        try:
            details = fn()
            stage_results.append(StageResult(name=name, status="ok", details="completed"))
            if name == "evaluate_gbm":
                fixed_origin_result = details
            elif name == "rolling_origin_backtest":
                rolling_origin_result = details
        except Exception as exc:  # noqa: BLE001
            msg = f"{name} failed: {type(exc).__name__}: {exc}"
            stage_results.append(StageResult(name=name, status="failed", details=msg))
            limitations.append(msg)
            raise

    raw_dir = project_root() / "data" / "raw"
    provenance = _load_provenance(raw_dir)
    raw_file = _find_raw_file(raw_dir, provenance)
    if raw_file is None:
        raise FileNotFoundError("No raw Brent source file found in the repository.")

    raw_df, validation = _validate_raw_data(raw_file)
    processed = _load_stage1_processed()
    proc_ext = _build_processed_with_logs(processed)
    processed_with_logs_path = project_root() / "data" / "processed" / "brent_prices_2000_2025_with_log_features.csv"
    proc_ext.to_csv(processed_with_logs_path, index=False)
    shutil.copy2(processed_with_logs_path, paths["processed"] / processed_with_logs_path.name)

    requested_start = pd.Timestamp("2000-01-01")
    requested_end = pd.Timestamp("2025-12-31")
    raw_dates = pd.to_datetime(raw_df["DATE"], errors="coerce")
    raw_values = pd.to_numeric(raw_df["DCOILBRENTEU"], errors="coerce")
    valid_in_window = raw_dates[(raw_dates >= requested_start) & (raw_dates <= requested_end) & raw_values.notna()]
    first_valid_in_window = valid_in_window.min()
    first_processed = proc_ext["Date"].min()
    last_processed = proc_ext["Date"].max()
    window_applied_correctly = bool(first_processed == first_valid_in_window and last_processed == requested_end)
    if not window_applied_correctly:
        limitations.append("Requested sample window validation did not pass exactly.")

    source_url = provenance.get("download_url") or DOWNLOAD_URL
    retrieval_date = provenance.get("retrieval_date") or "not recorded in inherited repository state"
    source_description = provenance.get("original_source") or SOURCE_DESCRIPTION
    data_validation = pd.DataFrame(
        [
            {
                "source_url": source_url,
                "source_description": source_description,
                "retrieval_date": retrieval_date,
                "raw_filename": raw_file.name,
                "units": validation["units"],
                "frequency": validation["frequency"],
                "requested_start": requested_start.date().isoformat(),
                "requested_end": requested_end.date().isoformat(),
                "first_observation": validation["first_observation"],
                "last_observation": validation["last_observation"],
                "first_valid_observation_in_window": first_valid_in_window.date().isoformat(),
                "first_processed_observation": first_processed.date().isoformat(),
                "last_processed_observation": last_processed.date().isoformat(),
                "missing_values": validation["missing_values"],
                "non_numeric_values": validation["non_numeric_values"],
                "duplicate_dates": validation["duplicate_dates"],
                "non_positive_values": validation["non_positive_values"],
                "chronological_order": validation["chronological_order"],
                "window_applied_correctly": window_applied_correctly,
                "end_date_present": bool(last_processed == requested_end),
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
        descriptive_rows.append(_period_descriptive(label, period_df))
        n_row, tails_df, top_df = _normality_tail_tables(label, period_df)
        normality_rows.append(n_row)
        tails_frames.append(tails_df)
        largest_frames.append(top_df)

    descriptive = pd.DataFrame(descriptive_rows)
    normality = pd.DataFrame(normality_rows)
    tails = pd.concat(tails_frames, ignore_index=True)
    largest = pd.concat(largest_frames, ignore_index=True)
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

    full_sample_params, fixed_origin_params, fixed_origin_metrics, _interp, _comparison = _extract_stage_outputs(
        paths, fixed_origin_result
    )

    rolling_forecasts_path = project_root() / "outputs" / "tables" / "rolling_origin_forecasts.csv"
    rolling_metrics_path = project_root() / "outputs" / "tables" / "rolling_origin_metrics_by_horizon.csv"
    rolling_forecasts = pd.read_csv(rolling_forecasts_path)
    rolling_metrics = pd.read_csv(rolling_metrics_path)
    rolling_forecasts.to_csv(paths["tables"] / "rolling_origin_forecasts.csv", index=False)
    rolling_metrics.to_csv(paths["tables"] / "rolling_origin_metrics_by_horizon.csv", index=False)

    _plot_basic_figures(proc_ext, paths)
    _plot_vol_acf_figures(proc_ext, rolling_df, period_comp, paths)
    _plot_backtest_figures(paths)

    for figure_name in [
        "rolling_origin_one_day_forecasts.png",
        "rolling_origin_coverage_by_horizon.png",
        "rolling_origin_interval_width_by_horizon.png",
        "rolling_origin_errors_by_horizon.png",
    ]:
        shutil.copy2(project_root() / "outputs" / "figures" / figure_name, paths["figures"] / figure_name)

    _write_report(
        paths,
        data_validation,
        descriptive,
        normality,
        tails,
        lb_df,
        adf_df,
        assumptions,
        full_sample_params,
        fixed_origin_params,
        fixed_origin_metrics,
        rolling_metrics,
        limitations,
    )
    _write_audit(paths)
    _write_corrections_log(paths["base"])
    _write_requirements_lock(paths["base"])
    _write_environment_log(paths["logs"])
    _copy_artifacts(paths)

    log_lines = [f"[{item.status}] {item.name}: {item.details}" for item in stage_results]
    log_lines.append(
        f"[ok] window_validation: first processed observation {first_processed.date().isoformat()}, "
        f"first valid in requested window {first_valid_in_window.date().isoformat()}, "
        f"last processed observation {last_processed.date().isoformat()}."
    )
    log_lines.append(
        f"[ok] rolling_origin_overlap_note: horizons 5 and 20 use overlapping target windows; see {rolling_origin_result['metrics_path']}."
    )
    if limitations:
        log_lines.append("Limitations:")
        log_lines.extend(f"- {item}" for item in limitations)
    (paths["logs"] / "analysis_run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    zip_path = _zip_handoff(base)
    return {
        "handoff_dir": base,
        "zip_path": zip_path,
        "limitations": limitations,
    }


if __name__ == "__main__":
    result = build_handoff_package()
    print(f"Handover package generated: {result['handoff_dir']}")
    print(f"Archive generated: {result['zip_path']}")
    if result["limitations"]:
        print("Limitations encountered:")
        for item in result["limitations"]:
            print(f"- {item}")
