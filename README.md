# Portfolio Risk & Performance Analyzer

**Python | Quantitative Finance | Portfolio Risk | Benchmark Analysis**

A Python-based portfolio risk and performance analysis project comparing an **equal-weight portfolio of 40 equities** with **SPY (an exchange-traded fund tracking the S&P 500)**. The main analysis uses adjusted historical prices, monthly rebalancing assumptions, risk-adjusted performance metrics, and explicit missing-data checks.

> **Research disclaimer:** This is a **retrospective analysis of a stock basket selected after the sample began**, not a point-in-time investable backtest. The results have selection/survivorship bias and are not evidence of future performance or investment advice.

## Highlights

- Analyze **40 stocks at 2.5% target weight each**, assuming monthly rebalancing.
- Compare portfolio **gross** and **net (before trading costs)** monthly returns with SPY.
- Calculate **CAGR, annualized volatility, Sharpe ratio, monthly-sampled maximum drawdown, beta, tracking error, downside capture, historical VaR, and Expected Shortfall**.
- Validate weights and price data; produce a coverage report and avoid filling missing returns with fabricated observations.
- Export CSV reports and a growth-of-$1 chart.
- Run locally with Yahoo Finance downloads or an offline monthly-price CSV.

## Reported backtest results

**Analysis window:** November 2025–October 2026 (**12 monthly returns**). Prices are fetched starting in October 2025 to establish the first monthly return. In the reported run, **all 40 holdings had complete data; none were excluded**.

The following numbers are from a successful local run of `portfolio_analyzer_fixed.py` (user-provided terminal output), not from an independently reproduced execution in this README.

| Metric | Portfolio Gross | Portfolio Net (before trading costs) | SPY |
|:--|--:|--:|--:|
| CAGR | 32.19% | **31.45%** | 12.00% |
| Annualized volatility | 25.30% | 25.29% | 13.30% |
| Sharpe ratio (fixed risk-free rate) | 1.0496 | **1.0274** | 0.5829 |
| Maximum drawdown (monthly sampled) | -10.62% | -10.79% | -5.76% |
| Beta vs SPY | 1.5828 | 1.5821 | 1.0000 |
| Tracking error vs SPY | 16.03% | 16.02% | 0.00% |
| Downside capture | 68.73% | 71.60% | 100.00% |
| Monthly 95% historical VaR (loss convention) | 5.47% | 5.51% | 2.79% |
| Monthly 95% Expected Shortfall (loss convention) | 6.52% | 6.57% | 4.94% |

**Interpretation:** The portfolio showed a higher return and Sharpe ratio than SPY over this specific 12-month sample, but also substantially higher volatility, beta, and monthly-sampled drawdown. This does **not** establish persistent outperformance. With only 12 monthly observations, estimated 95% VaR and Expected Shortfall are especially sensitive to individual observations.

## Methodology and assumptions

| Item | Assumption |
|:--|:--|
| Holdings | 40 equities listed in `PORTFOLIO` in `portfolio_analyzer_fixed.py` |
| Initial/target weights | 2.5% each (total 100%) |
| Portfolio mechanics | Constant target weights every month (implicit monthly rebalancing) |
| Benchmark | SPY |
| Price data | `yfinance` adjusted closing prices; sampled at month-end |
| Return observations | November 2025 through October 2026 |
| Annual risk-free rate | Fixed illustrative **4.5%**, converted to an effective monthly rate; **not** a historical Treasury-bill series |
| Annual management fee | 0.50% |
| Annual custody fee | 0.06% |
| Net return definition | Gross return adjusted by the combined fee factor; **trading costs, bid-ask spreads, slippage, taxes, and market impact are excluded** |
| Annualization | Monthly volatility × √12; CAGR based on compounded monthly returns |
| VaR / Expected Shortfall | Empirical monthly lower-tail measures; positive values denote estimated loss magnitude |
| Drawdown | Measured from **monthly-sampled** cumulative returns, not daily intramonth peaks/troughs |

### Missing-data policy

The main script checks for complete monthly returns for all holdings and SPY. Its default policy is **strict**: missing benchmark data or missing holding returns causes the analysis to fail rather than silently filling data.

You may explicitly choose `--missing-policy drop-assets` to remove holdings with missing monthly returns and renormalize the surviving weights. **This changes the portfolio being analyzed and can further bias comparisons.** Inspect the coverage and effective-weight CSVs before interpreting such a run.

## Installation

Requires a recent Python 3 version and internet access for live price downloads.

```bash
# From the repository root
python3 -m pip install -r requirements.txt
```

## Run the 40-stock analysis

```bash
python3 portfolio_analyzer_fixed.py
```

Optional: exclude incomplete-history holdings rather than stopping:

```bash
python3 portfolio_analyzer_fixed.py --missing-policy drop-assets
```

Optional: use an offline CSV of adjusted **monthly prices** with dates in the first column and ticker symbols as column headers:

```bash
python3 portfolio_analyzer_fixed.py --prices-csv monthly_prices.csv
```

Optional: select an output directory:

```bash
python3 portfolio_analyzer_fixed.py --output-dir outputs
```

By default, the main script writes the following files into `outputs/`:

| File | Description |
|:--|:--|
| `VIE_monthly_adjusted_prices.csv` | Month-end adjusted prices |
| `VIE_price_coverage.csv` | Missing-price coverage by ticker |
| `VIE_monthly_stock_returns.csv` | Per-stock monthly returns |
| `VIE_missing_price_counts.csv` | Missing return counts by month |
| `VIE_effective_weights.csv` | Actual weights after the selected missing-data policy |
| `VIE_monthly_returns_vs_SPY.csv` | Gross, net, and SPY monthly returns |
| `VIE_growth_of_one_dollar.csv` | Compounded growth series |
| `VIE_performance_metrics.csv` | Risk and performance metrics |
| `VIE_vs_SPY_monthly.png` | Growth-of-$1 chart |

The chart is generated when the analysis runs; it is **not** the same artifact as the legacy `portfolio_report.png` in the repository root.

## Testing

To run the unit tests from the repository root, with `test_fixed.py` present:

```bash
python3 test_fixed.py
```

A reported local run completed **7 tests with `OK`**. Unit tests check selected calculation and validation behaviors; they do not constitute an independent verification of Yahoo Finance data or of the backtest's investability.

## Legacy five-stock example

`portfolio_analyzer.py` is the original **daily-frequency** five-stock example. It demonstrates basic portfolio risk metrics and creates `portfolio_report.png` and `portfolio_summary.csv`:

```bash
python3 portfolio_analyzer.py
```

Its historical results, settings, and frequency should **not** be mixed with the primary 40-stock monthly SPY comparison. The older `portfolio_report.png` belongs to this example.

## Limitations

1. **Retrospective selection and survivorship bias:** Holdings were chosen after the analysis period began, so the results do not represent a point-in-time executable strategy.
2. **Short observation period:** Twelve monthly returns are insufficient for robust conclusions, especially for tail-risk statistics and stability of beta/Sharpe estimates.
3. **Costs:** The modeled net returns account for management and custody fees but not trading costs, taxes, turnover, or implementation constraints.
4. **Fixed assumptions:** Weights rebalance monthly and the 4.5% risk-free rate is illustrative rather than time-varying.
5. **Data dependency:** Yahoo Finance data can be corrected or revised, affecting reproducibility. Run dates and dependency versions can change results.
6. **Monthly sampling:** Maximum drawdown may understate intramonth losses.

## Potential extensions

- Point-in-time asset selection with genuine out-of-sample validation.
- Turnover estimation and explicit transaction-cost modeling.
- Longer evaluation periods where each security has valid history.
- Confidence intervals and sensitivity analysis for risk metrics.
- Automated tests in GitHub Actions and dependency locking.

## Why this project

This project demonstrates applied Python, financial data engineering, quantitative risk measurement, and responsible interpretation of historical performance. It emphasizes transparent assumptions and validation rather than treating a high retrospective return as proof of a superior investment strategy.
