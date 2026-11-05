"""Download fixed 2024 AAPL history from Yahoo; save original response and provenance.

Usage: python3 download_data.py
Network/provider availability is required. Later downloads may contain revisions.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
start=int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp())
end=int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp())
URL=f'https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1={start}&period2={end}&interval=1d&events=div%2Csplits'

if __name__=='__main__':
    request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read()
    parsed=json.loads(raw)
    if parsed['chart'].get('error') or not parsed['chart'].get('result'):raise ValueError('Provider returned no data')
    directory=ROOT/'data';directory.mkdir(exist_ok=True)
    (directory/'aapl_2024_raw.json').write_bytes(raw)
    manifest=dict(ticker='AAPL',provider='Yahoo Finance',url=URL,
                  requested_start='2024-01-01',requested_end_exclusive='2025-01-01',
                  retrieved_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    (directory/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
