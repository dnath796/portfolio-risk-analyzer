#!/usr/bin/env python3
"""VIE: PRINT the entire historical report directly in your Terminal.

Run:
    python3 -m pip install yfinance pandas numpy
    python3 -u vie_console_backtest.py

No CSV or Excel outputs are created.
Uses today's 40 selected tickers retrospectively, not a point-in-time
strategy simulation. Historical eligibility depends on available prices.
"""
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    'ADBE','GOOGL','APH','ADI','APA','AMAT','APP','ALAB','BKNG','AVGO',
    'COF','CAT','CRBG','CRDO','EXPE','FIGR','YMM','KLAC','KGS','MGNI',
    'MTCH','MU','NFLX','NVDA','NXPI','ONON','PAGS','PDD','PR','RDDT',
    'RYAN','SNDK','STX','SM','SYF','TAL','VRT','VST','WDC','ZTO',
]
START = '2015-12-01'
END_EXCLUSIVE = '2026-11-01'
RISK_FREE = 0.045
ANNUAL_MGMT_FEE = 0.005
ANNUAL_CUSTODY_FEE = 0.0006
BENCHMARK = 'SPY'
WINDOWS = [('1-Year',12),('3-Year',36),('5-Year',60),('10-Year',120)]

def fetch_returns():
    symbols = TICKERS + [BENCHMARK]
    print('Downloading adjusted closing prices for 40 stocks + SPY...', flush=True)
    raw = yf.download(symbols, start=START, end=END_EXCLUSIVE,
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError('No valid multi-ticker data. Check Yahoo Finance connection.')
    if 'Close' not in raw.columns.get_level_values(0):
        raise RuntimeError('Yahoo Finance returned no Close prices.')
    prices = raw['Close'].reindex(columns=symbols)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.resample('ME').last()
    ret = prices.pct_change(fill_method=None).loc['2016-01-31':'2026-10-31']
    active = ret[TICKERS].notna().sum(axis=1)
    gross = ret[TICKERS].mean(axis=1, skipna=True).where(active > 0)
    monthly_fee_factor = (1 - ANNUAL_MGMT_FEE - ANNUAL_CUSTODY_FEE)**(1/12)
    net = (1 + gross) * monthly_fee_factor - 1
    df = pd.DataFrame({'VIE':net, 'SPY':ret[BENCHMARK],
                       'VIE Gross':gross, 'Stocks':active}).dropna(subset=['VIE','SPY'])
    if df.empty:
        raise RuntimeError('No dates with both VIE and SPY returns.')
    if not df.index.to_period('M').equals(pd.period_range(df.index.min(),df.index.max(),freq='M')):
        raise RuntimeError('Missing calendar months: cannot annualize reliably.')
    df.index = df.index.to_period('M')
    return df

def safe_ratio(num,den):
    return float(num/den) if np.isfinite(num) and np.isfinite(den) and abs(den)>1e-12 else np.nan

def compounded_return(r):
    return float(np.expm1(np.log1p(r).sum())) if len(r) else np.nan

def cagr(r):
    return (1 + compounded_return(r))**(12/len(r))-1 if len(r) else np.nan

def annual_vol(r):
    return float(r.std(ddof=1)*np.sqrt(12)) if len(r)>1 else np.nan

def max_dd(r):
    wealth=np.r_[1.0, np.cumprod(1 + np.asarray(r, dtype=float))]
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())

def drawdown_timing(r):
    w=np.r_[1.0, np.cumprod(1 + np.asarray(r, dtype=float))]
    d=w/np.maximum.accumulate(w)-1
    trough=int(np.argmin(d)); peak=int(np.argmax(w[:trough+1]))
    if trough==0 or np.isclose(d[trough],0): return (0,0)
    recovery=np.flatnonzero(w[trough+1:] >= w[peak])
    recovery_months=int(recovery[0]+1) if len(recovery) else np.nan
    return (trough-peak, recovery_months)

def relation_metrics(p,b,rf):
    if len(p)<2: return dict.fromkeys(['Correlation','R-Squared','Beta','Alpha','Tracking Error','Information Ratio',
         'Batting Average','Average Excess Outperformance','Average Excess Underperformance','Excess Return Gain/Loss Ratio'], np.nan)
    pr,br=np.asarray(p,dtype=float),np.asarray(b,dtype=float)
    excess=pr-br
    corr=float(np.corrcoef(pr,br)[0,1]) if np.std(pr)>0 and np.std(br)>0 else np.nan
    beta=safe_ratio(np.cov(pr,br,ddof=1)[0,1],np.var(br,ddof=1))
    rf_m=(1+rf)**(1/12)-1
    alpha=12*(np.mean(pr)-rf_m-beta*(np.mean(br)-rf_m)) if np.isfinite(beta) else np.nan
    te=float(np.std(excess,ddof=1)*np.sqrt(12))
    up=excess[excess>0]; down=excess[excess<0]
    return {'Correlation':corr, 'R-Squared':corr**2 if np.isfinite(corr) else np.nan,
        'Beta':beta,'Alpha':alpha,'Tracking Error':te,
        'Information Ratio':safe_ratio(12*np.mean(excess),te),
        'Batting Average':float(np.mean(excess>0)),
        'Average Excess Outperformance':float(np.mean(up)) if len(up) else np.nan,
        'Average Excess Underperformance':float(np.mean(down)) if len(down) else np.nan,
        'Excess Return Gain/Loss Ratio':safe_ratio(np.mean(up),abs(np.mean(down))) if len(up) and len(down) else np.nan}

def capture(p,b,condition):
    p=np.asarray(p); b=np.asarray(b)
    x=b>0 if condition=='up' else b<0
    return 100*safe_ratio(np.mean(p[x]),np.mean(b[x])) if x.any() else np.nan

def absolute_metrics(r,rf,benchmark=None):
    r=np.asarray(r,dtype=float)
    positive=r[r>0]; negative=r[r<0]
    rf_m=(1+rf)**(1/12)-1
    up=down=np.nan
    if benchmark is not None:
        up=capture(r,benchmark,'up')
        down=capture(r,benchmark,'down')
    peak_to_trough, trough_to_peak=drawdown_timing(r)
    return {'Annualized Returns':cagr(r),'Annualized Volatility':annual_vol(pd.Series(r)),
        'Sharpe Ratio':safe_ratio((np.mean(r)-rf_m)*np.sqrt(12),np.std(r,ddof=1)) if len(r)>1 else np.nan,
        'Upside Capture':up,'Downside Capture':down,
        'Frequency of Positive Returns':np.mean(r>0),
        'Mean Positive Return':np.mean(positive) if len(positive) else np.nan,
        'Frequency of Negative Returns':np.mean(r<0),
        'Mean Negative Return':np.mean(negative) if len(negative) else np.nan,
        'Maximum Drawdown':max_dd(r),
        'Peak to Trough (months)':peak_to_trough,
        'Trough to Peak (months)':trough_to_peak,
        'Average Annual Turnover':np.nan}

def table_annual(df,start,end):
    rows=[]
    for year in range(start,end+1):
        s=df[df.index.year==year]
        full=len(s)==12 and set(s.index.month)==set(range(1,13))
        rows.append({'Calendar Year':year,'Backtest Results':compounded_return(s.VIE) if full else np.nan,
                     'S&P 500':compounded_return(s.SPY) if full else np.nan,
                     'Months Available':len(s), 'Status':'Full year' if full else 'N/A: fewer than 12 months'})
    return pd.DataFrame(rows)

def make_absolute_table(df,rf):
    rows=[]
    for label,n in WINDOWS + [('Since Inception',None)]:
        enough=n is None or len(df)>=n
        section=df.tail(n) if enough and n else (df if enough else None)
        if section is None:
            for metric in ['Annualized Returns','Annualized Volatility','Sharpe Ratio','Upside Capture',
                   'Downside Capture','Frequency of Positive Returns','Mean Positive Return',
                   'Frequency of Negative Returns','Mean Negative Return','Maximum Drawdown',
                   'Peak to Trough (months)','Trough to Peak (months)','Average Annual Turnover']:
                rows.append({'Period':label,'Metric':metric,'Backtest Results':np.nan,'S&P 500':np.nan})
        else:
            pm=absolute_metrics(section.VIE,rf,section.SPY)
            bm=absolute_metrics(section.SPY,rf,section.SPY)
            for metric in pm:
                rows.append({'Period':label,'Metric':metric,'Backtest Results':pm[metric], 'S&P 500':bm[metric]})
    return pd.DataFrame(rows)

def make_relative_table(df,rf):
    rows=[]
    names=['Correlation','R-Squared','Beta','Alpha','Tracking Error','Information Ratio',
       'Batting Average','Average Excess Outperformance','Average Excess Underperformance',
       'Excess Return Gain/Loss Ratio']
    for label,n in WINDOWS + [('Since Inception',None)]:
        enough=n is None or len(df)>=n
        section=df.tail(n) if enough and n else (df if enough else None)
        m=relation_metrics(section.VIE,section.SPY,rf) if section is not None else dict.fromkeys(names,np.nan)
        for metric in names:
            rows.append({'Period':label,'Metric':metric,'Backtest Results':m[metric]})
    return pd.DataFrame(rows)

def make_up_down(df,rf):
    rows=[]
    for condition, heading in [('up','S&P 500 Up Months'),('down','S&P 500 Down Months')]:
        s=df[df.SPY > 0] if condition=='up' else df[df.SPY < 0]
        rel=relation_metrics(s.VIE,s.SPY,rf) if len(s)>1 else {}
        cap=capture(s.VIE,s.SPY,condition) if len(s) else np.nan
        # For up months: frequency VIE is positive; for down months: frequency VIE is negative
        freq=(s.VIE>0).mean() if condition=='up' and len(s) else ((s.VIE<0).mean() if len(s) else np.nan)
        mapping={'Frequency of Positive Returns' if condition=='up' else 'Frequency of Negative Returns':freq,
                 'Upside Capture' if condition=='up' else 'Downside Capture':cap}
        mapping.update({k:rel.get(k,np.nan) for k in ['Correlation','Beta','Alpha','Batting Average',
          'Average Excess Outperformance','Average Excess Underperformance','Excess Return Gain/Loss Ratio']})
        for k,v in mapping.items():
            rows.append({'Market Condition':heading,'Metric':k,'Backtest Results':v,'Benchmark Months':len(s)})
    return pd.DataFrame(rows)


def fmt(metric,value):
    if pd.isna(value): return 'N/A'
    if metric in ('Months','Peak to Trough (months)','Trough to Peak (months)','Benchmark Months'):
        return f'{value:.0f}'
    if metric in ('Beta','Sharpe Ratio','Information Ratio','Excess Return Gain/Loss Ratio'):
        return f'{value:.3f}'
    if metric in ('Correlation','R-Squared'):
        return f'{value:.3f}'
    if 'Capture' in metric: return f'{value:.2f}%'
    if metric in ('Annualized Returns','Annualized Volatility','Alpha','Tracking Error',
                  'Frequency of Positive Returns','Frequency of Negative Returns',
                  'Mean Positive Return','Mean Negative Return','Maximum Drawdown',
                  'Batting Average','Average Excess Outperformance',
                  'Average Excess Underperformance','Average Annual Turnover'):
        return f'{value * 100:.2f}%'
    return f'{value:.4f}'

def show_table(title,table,metric_col='Metric',value_cols=('Backtest Results','S&P 500'),group_col=None):
    print('\n' + '='*92)
    print(title)
    print('='*92)
    out=table.copy()
    for col in value_cols:
        if col in out:
            out[col]=[fmt(str(row[metric_col]), row[col]) for _,row in out.iterrows()]
    print(out.to_string(index=False, na_rep='N/A', max_rows=None))

def main():
    df=fetch_returns()
    print('\n'+'='*92)
    print('VIE vs SPY | ACTUAL TESTED MONTHS')
    print('='*92)
    print(f'First month: {df.index.min()} | Last month: {df.index.max()} | Months: {len(df)}')
    print(f'Active stocks: first={int(df.Stocks.iloc[0])}, last={int(df.Stocks.iloc[-1])}; range={int(df.Stocks.min())}-{int(df.Stocks.max())}')
    print('VIE = net after assumed 0.50% management and 0.06% custody fees; excludes trading costs.')
    print('SPY = dividend-adjusted ETF proxy for S&P 500, not official index returns.')
    print('\nMONTHLY RETURNS (%):')
    month=df[['VIE','SPY','VIE Gross','Stocks']].copy()
    for c in ['VIE','SPY','VIE Gross']: month[c] = month[c].map(lambda v:f'{v*100:8.2f}%')
    print(month.to_string(max_rows=None))

    yr=table_annual(df,2006,2026)
    for c in ['Backtest Results','S&P 500']:
        yr[c]=yr[c].map(lambda v: 'N/A' if pd.isna(v) else f'{v*100:.2f}%')
    print('\n'+'='*92+'\nCALENDAR-YEAR RETURNS (complete calendar years only)\n'+'='*92)
    print(yr.to_string(index=False, max_rows=None))

    absolute=make_absolute_table(df,RISK_FREE)
    for period,subset in absolute.groupby('Period',sort=False):
        show_table(f'ABSOLUTE PERFORMANCE | {period}', subset)

    relative=make_relative_table(df,RISK_FREE)
    for period,subset in relative.groupby('Period',sort=False):
        show_table(f'RELATIVE PERFORMANCE VS SPY | {period}',subset,value_cols=('Backtest Results',))

    updown=make_up_down(df,RISK_FREE)
    for condition,subset in updown.groupby('Market Condition',sort=False):
        show_table(condition.upper(),subset,value_cols=('Backtest Results','Benchmark Months'))

    print('\n'+'='*92+'\nMETHODOLOGY NOTES\n'+'='*92)
    print('* N/A: insufficient history, missing qualifying months, or unavailable turnover data.')
    print('* Selected stocks are the CURRENT VIE list. This is NOT a point-in-time screen backtest.')
    print('* Portfolio reweights equally among names with monthly returns; newly listed stocks enter later.')
    print('* Bias: hindsight selection, survivorship, IPO availability, and potential sector drift.')
    print('* Sharpe and alpha assume a constant 4.5% annual risk-free rate.')
    print('* Max drawdown uses month-end returns; interim daily declines are not captured.')
    print('* Capture uses arithmetic average portfolio/benchmark returns in benchmark up/down months.')
    print('* Annual turnover is N/A without actual holdings and traded share/weight records.')
    print('* No files created: all results above were printed to this console.')

if __name__ == '__main__':
    main()
