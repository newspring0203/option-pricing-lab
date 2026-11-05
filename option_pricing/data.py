"""Validate a saved Yahoo chart response; never silently fill missing prices."""
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np


def parse_prices(path):
    raw=json.loads(Path(path).read_text())
    if raw['chart'].get('error'):raise ValueError(raw['chart']['error'])
    result=raw['chart']['result'][0]
    meta=result['meta']
    if meta['symbol']!='AAPL' or meta['currency']!='USD':raise ValueError('Unexpected ticker/currency')
    timestamps=result['timestamp']
    close=result['indicators']['quote'][0]['close']
    adjusted=result['indicators']['adjclose'][0]['adjclose']
    if not len(timestamps)==len(close)==len(adjusted):raise ValueError('Unequal input column lengths')
    rows=[]
    for ts,c,a in zip(timestamps,close,adjusted):
        if c is None or a is None:raise ValueError('Missing close/adjusted close')
        if not np.isfinite([c,a]).all() or min(c,a)<=0:raise ValueError('Invalid price')
        date=datetime.fromtimestamp(ts,ZoneInfo(meta['exchangeTimezoneName'])).date().isoformat()
        rows.append(dict(date=date,close=float(c),adjusted_close=float(a)))
    dates=[row['date'] for row in rows]
    if dates!=sorted(set(dates)):raise ValueError('Dates must be unique and ordered')
    if not all('2024-01-01'<=d<'2025-01-01' for d in dates):raise ValueError('Unexpected date range')
    return rows,dict(currency='USD',exchange_timezone=meta['exchangeTimezoneName'],
                     split_events=len(result.get('events',{}).get('splits',{})),
                     dividend_events=len(result.get('events',{}).get('dividends',{})))


def estimate_volatility(rows,annualization=252):
    if not isinstance(annualization,int) or isinstance(annualization,bool) or annualization<=0:
        raise ValueError('annualization must be a positive integer')
    if len(rows)<3:raise ValueError('At least three observations required')
    prices=np.asarray([r['adjusted_close'] for r in rows],dtype=float)
    if not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('Invalid adjusted prices')
    returns=np.diff(np.log(prices))
    return returns,float(returns.std(ddof=1)*np.sqrt(annualization))


def save_csv(rows,path):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['date','close','adjusted_close'])
        writer.writeheader();writer.writerows(rows)


def verify_source(path,manifest_path):
    manifest=json.loads(Path(manifest_path).read_text())
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest!=manifest['sha256']:raise ValueError('Raw source does not match manifest SHA-256')
    return digest
