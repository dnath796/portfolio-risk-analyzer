#!/usr/bin/env python3
"""VIE: Export historical portfolio report to a formatted Excel workbook.

Run:
    python3 -m pip install yfinance pandas numpy
    python3 -u vie_console_backtest.py

Outputs: VIE_Backtest_Report.xlsx (monthly returns, annual returns, risk statistics).
Uses today's 40 selected tickers retrospectively, not a point-in-time
strategy simulation. Historical eligibility depends on available prices.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

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

def export_excel(df, path='VIE_Backtest_Report.xlsx'):
    """Write all RFP report sections to worksheets, retaining numeric values."""
    annual = table_annual(df, 2006, 2026)
    absolute = make_absolute_table(df, RISK_FREE)
    relative = make_relative_table(df, RISK_FREE)
    up_down = make_up_down(df, RISK_FREE)
    monthly = df.reset_index().rename(columns={'index': 'Month', 'VIE': 'VIE Net Return',
                                                'SPY': 'SPY Return', 'VIE Gross': 'VIE Gross Return',
                                                'Stocks': 'Eligible Stocks'})
    monthly = monthly.rename(
    columns={monthly.columns[0]: "Month"})
    monthly["Month"] = monthly["Month"].astype(str)
    coverage = pd.DataFrame([
        ('First observed month', str(df.index.min())),
        ('Last observed month', str(df.index.max())),
        ('Number of monthly returns', int(len(df))),
        ('Eligible stocks first month', int(df.Stocks.iloc[0])),
        ('Eligible stocks last month', int(df.Stocks.iloc[-1])),
        ('Minimum eligible stocks', int(df.Stocks.min())),
        ('Maximum eligible stocks', int(df.Stocks.max())),
        ('Benchmark', 'SPY adjusted prices (S&P 500 proxy)'),
        ('Management fee assumption', ANNUAL_MGMT_FEE),
        ('Custody fee assumption', ANNUAL_CUSTODY_FEE),
        ('Annual risk-free assumption', RISK_FREE),
        ('Methodology', 'Equal-weighted current-stock retrospective simulation'),
        ('Important limitation', 'Not a point-in-time VIE quantitative screen backtest'),
        ('Transaction costs', 'Not included'),
        ('Drawdown measurement', 'Month-end sampled'),
        ('Turnover', 'Unavailable without holdings/trade history'),
    ], columns=['Field', 'Value'])

    data = {
        'Overview': coverage,
        'Monthly Returns': monthly,
        'Calendar Year': annual,
        'Absolute Performance': absolute,
        'Relative Performance': relative,
        'Up Down Months': up_down,
    }
    with pd.ExcelWriter(path, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
        book = writer.book
        header = book.add_format({'bold': True, 'bg_color': '#183153', 'font_color': '#FFFFFF',
                                  'border': 0, 'text_wrap': True, 'valign': 'vcenter'})
        pct = book.add_format({'num_format':'0.00%;[Red](0.00%)'})
        number = book.add_format({'num_format':'0.000'})
        integer = book.add_format({'num_format':'0'})
        for name, data_frame in data.items():
            data_frame.to_excel(writer, sheet_name=name, index=False, na_rep='N/A')
            sheet = writer.sheets[name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(data_frame), len(data_frame.columns)-1)
            sheet.set_row(0, 30, header)
            sheet.set_column(0, 0, 29)
            sheet.set_column(1, len(data_frame.columns)-1, 23)
            if name == 'Overview':
                sheet.set_column(0, 0, 32)
                sheet.set_column(1, 1, 75)
                for row_index in (9,10,11):
                    sheet.set_row(row_index, None, pct)
            elif name == 'Monthly Returns':
                sheet.set_column(1, 3, 22, pct)
                sheet.set_column(4, 4, 18, integer)
            elif name == 'Calendar Year':
                sheet.set_column(1, 2, 24, pct)
            elif name == 'Absolute Performance':
                sheet.set_column(2, 3, 25, number)
                for idx,row in data_frame.iterrows():
                    metric=row['Metric']
                    if metric in ('Annualized Returns','Annualized Volatility',
                                   'Frequency of Positive Returns','Mean Positive Return',
                                   'Frequency of Negative Returns','Mean Negative Return',
                                   'Maximum Drawdown','Average Annual Turnover'):
                        sheet.set_row(idx+1, None, pct)
            elif name == 'Relative Performance':
                sheet.set_column(2, 2, 26, number)
                for idx,row in data_frame.iterrows():
                    if row['Metric'] in ('Alpha','Tracking Error','Batting Average',
                                         'Average Excess Outperformance','Average Excess Underperformance'):
                        sheet.set_row(idx+1, None, pct)
            elif name == 'Up Down Months':
                sheet.set_column(2, 2, 26, number)
                sheet.set_column(3, 3, 21, integer)
        # NOTE: capture ratios are stored as percentage points (e.g. 71.6, not .716).
        notes = book.add_worksheet('Methodology Notes')
        notes.set_column(0, 0, 115)
        notes.write(0, 0, 'VIE RETROSPECTIVE BACKTEST — IMPORTANT LIMITATIONS', header)
        notes_list = [
            "Today's 40 selected securities are used for all dates when prices are available; this introduces look-ahead and survivorship biases.",
            'New IPOs enter after sufficient price history. The number of holdings changes over time.',
            'No historical PEG, margins, leverage, FCF or growth screens are reconstructed.',
            'Portfolio is equal-weighted monthly among stocks with valid monthly total returns.',
            'SPY adjusted prices are used as an S&P 500 total return proxy, not an exact index series.',
            'Returns labeled VIE Net deduct modeled annual management + custody fees (0.56%), but not actual trading costs.',
            'Only full 12-month calendar years show annual returns. Insufficient 3-, 5-, 10-year windows show N/A.',
            'Alpha and Sharpe use a fixed 4.5% annual risk-free assumption, not historical monthly T-bill rates.',
            'Capture ratios use average monthly returns in SPY up/down months.',
            'Drawdowns are month-end sampled, not daily high-to-low.',
            'Average annual turnover cannot be calculated without a dated historical holdings/trade file.',
            'Calendar years 2006-2015 have no requested portfolio coverage and appear as N/A.',
        ]
        for i,note in enumerate(notes_list,2): notes.write(i,0,note)
    return path


def main():
    print('Fetching adjusted monthly prices and constructing returns...', flush=True)
    df = fetch_returns()
    print('Actual tested months: %s through %s (%d months)' %
          (df.index.min(),df.index.max(),len(df)))
    print('Eligible stocks: %d to %d per month' % (df.Stocks.min(), df.Stocks.max()))
    filename=export_excel(df)
    print('Saved Excel report: ' + str(Path(filename).resolve()))
    print('Worksheets: Overview, Monthly Returns, Calendar Year, Absolute Performance,')
    print('            Relative Performance, Up Down Months, Methodology Notes')

if __name__ == '__main__':
    main()
