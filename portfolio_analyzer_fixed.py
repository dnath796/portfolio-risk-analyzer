"""VIE retrospective 40-stock monthly backtest versus SPY.

Install: python3 -m pip install yfinance pandas numpy matplotlib
Run:     python3 -u portfolio_analyzer_fixed.py

Fetches dividend-adjusted monthly close prices, checks complete month-end data,
and exports CSV reports plus a PNG growth comparison. This tests the 2026
selected basket retrospectively, NOT a point-in-time 2016-2026 strategy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

PORTFOLIO = {
    "ADBE": .025, "GOOGL": .025, "APH": .025, "ADI": .025,
    "APA": .025, "AMAT": .025, "APP": .025, "ALAB": .025,
    "BKNG": .025, "AVGO": .025, "COF": .025, "CAT": .025,
    "CRBG": .025, "CRDO": .025, "EXPE": .025, "FIGR": .025,
    "YMM": .025, "KLAC": .025, "KGS": .025, "MGNI": .025,
    "MTCH": .025, "MU": .025, "NFLX": .025, "NVDA": .025,
    "NXPI": .025, "ONON": .025, "PAGS": .025, "PDD": .025,
    "PR": .025, "RDDT": .025, "RYAN": .025, "SNDK": .025,
    "STX": .025, "SM": .025, "SYF": .025, "TAL": .025,
    "VRT": .025, "VST": .025, "WDC": .025, "ZTO": .025,
}
BENCHMARK = "SPY"
START = "2025-10-01"  # preceding month to compute October 2025 return
END_EXCLUSIVE = "2026-11-01"  # stop after October 2026
FIRST_RETURN_MONTH = "2025-11-30"
LAST_RETURN_MONTH = "2026-10-31"

# 20 years data

# PORTFOLIO = {
#     "ADBE": 1/17,
#     "GOOGL": 1/17,
#     "APH": 1/17,
#     "ADI": 1/17,
#     "APA": 1/17,
#     "AMAT": 1/17,
#     "BKNG": 1/17,
#     "COF": 1/17,
#     "CAT": 1/17,
#     "EXPE": 1/17,
#     "KLAC": 1/17,
#     "MU": 1/17,
#     "NFLX": 1/17,
#     "NVDA": 1/17,
#     "STX": 1/17,
#     "SM": 1/17,
#     "WDC": 1/17,
# }
# BENCHMARK = "SPY"
# START = "2005-10-01"  # preceding month to compute October 2025 return
# END_EXCLUSIVE = "2026-11-01"  # stop after October 2026
# FIRST_RETURN_MONTH = "2005-11-30"
# LAST_RETURN_MONTH = "2026-10-31"


# PORTFOLIO = {
#     "ADBE": 1/26,
#     "GOOGL": 1/26,
#     "APH": 1/26,
#     "ADI": 1/26,
#     "APA": 1/26,
#     "AMAT": 1/26,
#     "BKNG": 1/26,
#     "AVGO": 1/26,
#     "COF": 1/26,
#     "CAT": 1/26,
#     "EXPE": 1/26,
#     "KLAC": 1/26,
#     "MGNI": 1/26,
#     "MTCH": 1/26,
#     "MU": 1/26,
#     "NFLX": 1/26,
#     "NVDA": 1/26,
#     "NXPI": 1/26,
#     "PR": 1/26,
#     "STX": 1/26,
#     "SM": 1/26,
#     "SYF": 1/26,
#     "TAL": 1/26,
#     "VST": 1/26,
#     "WDC": 1/26,
#     "ZTO": 1/26,
# }
# BENCHMARK = "SPY"
# START = "2016-10-01"  # preceding month to compute October 2016 return
# END_EXCLUSIVE = "2026-11-01"  # stop after October 2026
# FIRST_RETURN_MONTH = "2016-11-30"
# LAST_RETURN_MONTH = "2026-10-31"
# print("Number of stocks:", len(PORTFOLIO))
# print("Total weight:", sum(PORTFOLIO.values()))


ANNUAL_RISK_FREE = 0.045  # fixed illustrative assumption, not historical T-bill yield
MANAGEMENT_FEE = 0.0050
CUSTODY_FEE = 0.0006
OUT = Path(__file__).resolve().parent


def get_monthly_prices():
    tickers = list(PORTFOLIO) + [BENCHMARK]
    print(f"Downloading adjusted prices for {len(tickers)} symbols...", flush=True)
    raw = yf.download(tickers, start=START, end=END_EXCLUSIVE,
                      auto_adjust=True, progress=False, threads=False,
                      group_by="column")
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data; check network access and ticker symbols")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Expected a multi-ticker response from Yahoo Finance")
    if "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance response lacks the Close field")
    prices = raw["Close"].copy().reindex(columns=tickers).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    # Do not forward-fill missing prices; preserve actual coverage gaps.
    return prices.resample("ME").last()


def safe_cagr(r):
    return (1 + r).prod() ** (12 / len(r)) - 1 if len(r) else np.nan


def drawdown(r):
    wealth = np.r_[1., (1 + r).cumprod().to_numpy()]
    return np.min(wealth / np.maximum.accumulate(wealth) - 1)


def stats(r, benchmark):
    r, benchmark = r.align(benchmark, join="inner")
    rf_monthly = (1 + ANNUAL_RISK_FREE) ** (1 / 12) - 1
    active = r - benchmark
    downside = benchmark < 0
    vol = r.std(ddof=1)
    bench_var = benchmark.var(ddof=1)
    capture = (r[downside].mean() / benchmark[downside].mean() * 100
               if downside.any() and benchmark[downside].mean() != 0 else np.nan)
    return {
        "Months": len(r),
        "CAGR": safe_cagr(r),
        "Annual volatility": vol * np.sqrt(12),
        "Sharpe (fixed 4.5% RF)": (r.mean() - rf_monthly) / vol * np.sqrt(12) if vol > 0 else np.nan,
        "Maximum drawdown (monthly sampled)": drawdown(r),
        "Beta vs SPY": r.cov(benchmark) / bench_var if bench_var > 0 else np.nan,
        "Tracking error vs SPY": active.std(ddof=1) * np.sqrt(12),
        "Downside capture (%)": capture,
        "Monthly 95% historical VaR": -np.percentile(r, 5),
    }


def main():
    print("VIE monthly backtest starting...", flush=True)
    if not np.isclose(sum(PORTFOLIO.values()), 1):
        raise ValueError("Portfolio weights must sum to 1")

    prices = get_monthly_prices()
    prices.to_csv(OUT / "VIE_monthly_adjusted_prices.csv", float_format="%.8f")
    print(f"Saved monthly adjusted prices: {prices.shape[0]} month's", flush=True)

    # Explicit coverage report makes IPO and missing-price issues visible.
    missing = prices.isna().sum().rename("missing_month_end_prices").to_frame()
    missing["first_valid_month_end"] = [prices[t].first_valid_index() for t in prices]
    missing["last_valid_month_end"] = [prices[t].last_valid_index() for t in prices]
    missing.to_csv(OUT / "VIE_price_coverage.csv")

    monthly_returns = prices.pct_change(fill_method=None)
    first = pd.Timestamp(FIRST_RETURN_MONTH)
    last = pd.Timestamp(LAST_RETURN_MONTH)
    monthly_returns = monthly_returns.loc[first:last]
    monthly_returns.to_csv(OUT / "VIE_monthly_stock_returns.csv", float_format="%.8f")
    monthly_returns.isna().sum(axis=1).rename("missing_ticker_returns").to_csv(
        OUT / "VIE_missing_price_counts.csv")

    expected_months = pd.date_range(first, last, freq="ME")
    monthly_returns = monthly_returns.reindex(expected_months)
    incomplete = monthly_returns.isna().any(axis=1)
    if incomplete.any():
        bad = monthly_returns.index[incomplete].strftime("%Y-%m").tolist()
        print(f"{bad}")
        print("WARNING: Not every ticker has a valid monthly return in: " + ", ".join(bad), flush=True)
        print("Check VIE_price_coverage.csv and VIE_monthly_stock_returns.csv", flush=True)
        print("Stopping: no portfolio performance has been calculated from incomplete data.", flush=True)
        return

    stock_returns = monthly_returns[list(PORTFOLIO)]
    weights = pd.Series(PORTFOLIO)
    vie_gross = stock_returns.mul(weights, axis=1).sum(axis=1)
    spy = monthly_returns[BENCHMARK]

    # Convert annual management/custody expense into equivalent monthly drag.
    expense_multiplier = (1 - MANAGEMENT_FEE - CUSTODY_FEE) ** (1 / 12)
    vie_net = (1 + vie_gross) * expense_multiplier - 1
    comparison = pd.DataFrame({"VIE Gross": vie_gross, "VIE Net (before trading costs)": vie_net, "SPY": spy})
    comparison.index.name = "Month"
    comparison.to_csv(OUT / "VIE_monthly_returns_vs_SPY.csv", float_format="%.8f")

    growth = (1 + comparison).cumprod()
    growth.to_csv(OUT / "VIE_growth_of_one_dollar.csv", float_format="%.8f")

    metrics = pd.DataFrame({col: stats(comparison[col], comparison["SPY"])
                            for col in comparison}).T
    metrics.index.name = "Series"
    metrics.to_csv(OUT / "VIE_performance_metrics.csv", float_format="%.8f")

    fig, ax = plt.subplots(figsize=(10, 6))
    growth.plot(ax=ax)
    ax.set(title="VIE 40-stock retrospective portfolio vs SPY", ylabel="Growth of $1", xlabel="Month")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "VIE_vs_SPY_monthly.png", dpi=160)
    plt.close(fig)

    print("\nBacktest period:", comparison.index.min().date(), "to", comparison.index.max().date(), flush=True)
    print("Monthly observations:", len(comparison), flush=True)
    print("\nMetrics (returns are decimals):\n", metrics.round(4).to_string(), flush=True)
    print("\nCSV and PNG outputs saved in:", OUT, flush=True)
    print("NOTE: Current-stock retrospective test; not a point-in-time investment strategy backtest.", flush=True)


if __name__ == "__main__":
    main()
