"""Retrospective equal-weight 40-stock monthly portfolio comparison against SPY.

This is NOT a point-in-time tradable historical strategy: the stock list was
selected after the start of the sample (survivorship/selection bias).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
START = "2025-10-01"             # A preceding month is needed for Nov returns
END_EXCLUSIVE = "2026-11-01"    # Includes trading through October 2026
FIRST_RETURN_MONTH = "2025-11-30"
LAST_RETURN_MONTH = "2026-10-31"
ANNUAL_RISK_FREE = 0.045        # Illustrative constant; not historical T-bills
MANAGEMENT_FEE = 0.0050
CUSTODY_FEE = 0.0006


def validate_weights(weights: dict[str, float]) -> None:
    if not weights or any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be nonnegative, finite, and nonempty")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise ValueError("Portfolio weights must sum to 1")
    if BENCHMARK in weights:
        raise ValueError("Benchmark must not also be a portfolio holding")


def download_monthly_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with pip install -r requirements.txt") from exc
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False, group_by="column")
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Missing or unexpected Yahoo Finance multi-ticker data")
    if "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance response does not contain Close prices")
    prices = raw["Close"].reindex(columns=tickers).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    # Resample last observed trading close, WITHOUT filling an entirely empty month.
    return prices.resample("ME").last()


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price index must be dates")
    if prices.index.has_duplicates:
        raise ValueError("Price dates must be unique")
    prices = prices.sort_index().reindex(columns=tickers)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    return prices


def coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "missing_month_end_prices": prices.isna().sum(),
        "first_valid_month_end": [prices[c].first_valid_index() for c in prices],
        "last_valid_month_end": [prices[c].last_valid_index() for c in prices],
    }, index=prices.columns)


def monthly_return_table(prices: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    expected = pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="ME")
    if len(expected) == 0:
        raise ValueError("Empty analysis period")
    return returns.reindex(expected)


def select_complete_assets(returns: pd.DataFrame, weights: dict[str, float],
                           policy: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Strict fails on incomplete holdings; drop-assets renormalizes survivors.

    SPY must ALWAYS have complete data. No return or price is fabricated.
    """
    if returns[BENCHMARK].isna().any():
        missing = returns.index[returns[BENCHMARK].isna()].strftime('%Y-%m').tolist()
        raise ValueError(f"Benchmark {BENCHMARK} missing monthly returns: {missing}")
    bad = [ticker for ticker in weights if returns[ticker].isna().any()]
    if bad and policy == "strict":
        months = returns.index[returns[list(weights)].isna().any(axis=1)].strftime('%Y-%m').tolist()
        raise ValueError(
            f"Incomplete holding returns: {bad}; affected months: {months}. "
            "Inspect the coverage CSV or use --missing-policy drop-assets "
            "(changes basket composition and introduces selection bias)."
        )
    survivors = [ticker for ticker in weights if ticker not in bad]
    if not survivors:
        raise ValueError("No complete-history holdings remain")
    w = pd.Series({ticker: weights[ticker] for ticker in survivors}, dtype=float)
    w = w / w.sum()
    return returns[survivors + [BENCHMARK]], w, bad


def calculate_monthly_series(returns: pd.DataFrame, weights: pd.Series,
                             management_fee: float = MANAGEMENT_FEE,
                             custody_fee: float = CUSTODY_FEE) -> pd.DataFrame:
    if any(not (0 <= f < 1) for f in (management_fee, custody_fee)):
        raise ValueError("Fees must be between 0 and 1")
    if management_fee + custody_fee >= 1:
        raise ValueError("Combined annual fees must be below 100%")
    if returns.isna().any().any():
        raise ValueError("Return table contains missing values")
    # Constant-weight monthly rebalancing, ignoring turnover and trading costs.
    gross = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    fee_multiplier = (1 - management_fee - custody_fee) ** (1 / 12)
    net = (1 + gross) * fee_multiplier - 1
    return pd.DataFrame({"Portfolio Gross": gross,
                         "Portfolio Net (before trading costs)": net,
                         BENCHMARK: returns[BENCHMARK]})


def max_drawdown(monthly_returns: pd.Series) -> float:
    wealth = np.r_[1.0, (1 + monthly_returns).cumprod().to_numpy()]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))


def statistics(returns: pd.Series, benchmark: pd.Series,
               annual_rf: float = ANNUAL_RISK_FREE) -> dict:
    r, b = returns.align(benchmark, join="inner")
    if len(r) < 2 or r.isna().any() or b.isna().any():
        raise ValueError("At least two aligned, valid monthly returns required")
    if (r <= -1).any() or (b <= -1).any():
        raise ValueError("Returns cannot be -100% or lower for CAGR calculation")
    rf_month = (1 + annual_rf) ** (1 / 12) - 1
    vol = float(r.std(ddof=1))
    b_var = float(b.var(ddof=1))
    active = r - b
    down = b < 0
    threshold = float(np.quantile(r, .05))
    tail = r[r <= threshold]
    return {
        "Months": len(r),
        "CAGR": (1 + r).prod() ** (12 / len(r)) - 1,
        "Annual volatility": vol * np.sqrt(12),
        "Sharpe (fixed RF)": (r.mean() - rf_month) / vol * np.sqrt(12) if vol > 0 else np.nan,
        "Maximum drawdown (monthly sampled)": max_drawdown(r),
        "Beta vs SPY": r.cov(b) / b_var if b_var > 0 else np.nan,
        "Tracking error vs SPY": active.std(ddof=1) * np.sqrt(12),
        "Downside capture (%)": r[down].mean() / b[down].mean() * 100 if down.any() else np.nan,
        "Monthly 95% historical VaR (loss convention)": -threshold,
        "Monthly 95% Expected Shortfall (loss convention)": -float(tail.mean()),
    }


def run(prices: pd.DataFrame, out: Path, policy: str = "strict") -> pd.DataFrame:
    validate_weights(PORTFOLIO)
    if policy not in ("strict", "drop-assets"):
        raise ValueError("Unsupported missing-data policy")
    out.mkdir(parents=True, exist_ok=True)
    tickers = list(PORTFOLIO) + [BENCHMARK]
    prices = validate_prices(prices, tickers)
    prices.to_csv(out / "VIE_monthly_adjusted_prices.csv")
    coverage_report(prices).to_csv(out / "VIE_price_coverage.csv")
    returns = monthly_return_table(prices, FIRST_RETURN_MONTH, LAST_RETURN_MONTH)
    returns.to_csv(out / "VIE_monthly_stock_returns.csv")
    returns.isna().sum(axis=1).rename("missing_ticker_returns").to_csv(out / "VIE_missing_price_counts.csv")
    selected, weights, excluded = select_complete_assets(returns, PORTFOLIO, policy)
    weights.rename("effective_weight").to_csv(out / "VIE_effective_weights.csv")
    comparison = calculate_monthly_series(selected, weights)
    comparison.index.name = "Month"
    comparison.to_csv(out / "VIE_monthly_returns_vs_SPY.csv")
    growth = (1 + comparison).cumprod()
    growth.to_csv(out / "VIE_growth_of_one_dollar.csv")
    metrics = pd.DataFrame({col: statistics(comparison[col], comparison[BENCHMARK])
                            for col in comparison}).T
    metrics.index.name = "Series"
    metrics.to_csv(out / "VIE_performance_metrics.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    growth.plot(ax=ax)
    ax.set(title=f"Retrospective portfolio ({len(weights)} stocks) vs SPY",
           ylabel="Growth of $1", xlabel="Month")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "VIE_vs_SPY_monthly.png", dpi=160)
    plt.close(fig)
    print(f"Completed {len(comparison)} months with {len(weights)} holdings; excluded: {excluded}")
    print("Selection bias warning: holdings are selected retrospectively.")
    print(metrics.round(4).to_string())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-policy", choices=("strict", "drop-assets"), default="strict")
    parser.add_argument("--prices-csv", type=Path, help="Offline adjusted monthly price CSV (dates as first column)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    tickers = list(PORTFOLIO) + [BENCHMARK]
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    else:
        prices = download_monthly_prices(tickers, START, END_EXCLUSIVE)
    run(prices, args.output_dir, args.missing_policy)


if __name__ == "__main__":
    main()
