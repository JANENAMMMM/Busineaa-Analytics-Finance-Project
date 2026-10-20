"""Extended-hours price recency features. Bars are kept only when known_at < target 09:30."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,pyarrow.parquet as pq

HOLD=pd.Timestamp('2026-06-01')
CLOSE=pd.Timedelta(hours=16)
CUTOFF=pd.Timedelta(hours=9,minutes=30)
OUT=ROOT/'cache/oct12_price_recency.parquet'

def main():
    dev=pd.read_parquet(ROOT/'reports/oct10_extension/development.parquet',columns=['date','target','symbol'])
    assert dev.target.max()<HOLD
    cal=dev[['date','target']].drop_duplicates().sort_values('date').reset_index(drop=True)
    cal['start']=cal.date+CLOSE
    cal['cutoff']=cal.target+CUTOFF
    p=pq.read_table(ROOT/'dataset/price.parquet',
                    columns=['symbol','datetime','close','session','known_at'],
                    filters=[('known_at','<',HOLD+CUTOFF)]).to_pandas()
    p=p[p.session.isin(['post','pre'])].sort_values('datetime')
    # Attach each extended-hours bar to the overnight window that opens at the prior 16:00.
    i=np.searchsorted(cal.start.to_numpy(),p.datetime.to_numpy(),side='right')-1
    keep=i>=0;p=p.loc[keep].copy();i=i[keep]
    p['date']=cal.date.to_numpy()[i];p['target']=cal.target.to_numpy()[i]
    cut=cal.cutoff.to_numpy()[i]
    day=p.datetime.dt.normalize().to_numpy()
    on_window=((p.session.to_numpy()=='post')&(day==p.date.to_numpy()))|((p.session.to_numpy()=='pre')&(day==p.target.to_numpy()))
    p=p[on_window&(p.datetime.to_numpy()<cut)&(p.known_at.to_numpy()<cut)].copy()
    p['cutoff']=p.target+CUTOFF
    g=p.sort_values('datetime').groupby(['date','target','symbol'],sort=False)
    last=g.tail(1).set_index(['date','target','symbol'])
    out=dev.set_index(['date','target','symbol']).copy()
    out['price_age_min']=(last.cutoff-last.known_at).dt.total_seconds().div(60)
    out['has_pre']=p[p.session=='pre'].groupby(['date','target','symbol']).size().gt(0).reindex(out.index).fillna(False).astype(int)
    out['ext_bars']=g.size().reindex(out.index).fillna(0).astype(int)
    out=out.reset_index()
    assert out.price_age_min.dropna().ge(0).all()
    out.to_parquet(OUT,index=False)
    print(out.shape)
    print(out[['price_age_min','has_pre','ext_bars']].describe().round(3))
    print('age missing',int(out.price_age_min.isna().sum()),'has_pre=1',int(out.has_pre.sum()))

if __name__=='__main__':main()
