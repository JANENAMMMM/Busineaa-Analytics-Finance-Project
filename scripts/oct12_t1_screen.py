"""T1 screen: existing LLM direction on high-novelty days. fold1 train, 40 training symbols, no API calls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import binomtest

OUT=ROOT/'reports/followup_checks';OUT.mkdir(parents=True,exist_ok=True)
SPEC=json.loads((ROOT/'lab/folds.json').read_text(encoding='utf-8'))
GATE={'min_moved_with_direction':100,'min_disagree':50,'hit_min':0.58,'one_sided_p_lt':0.01,'disagree_hit_gt':0.5}
MIN_PRIOR_DAYS=20
HIGH_Q=0.8

def llm_direction():
    f=pd.read_parquet(ROOT/'cache/oct11_llm_full/full.parquet',columns=['date','target','symbol','raw'])
    j=f.raw.apply(lambda s:json.loads(s) if isinstance(s,str) else {})
    f['llm']=pd.to_numeric(j.str.get('direction'),errors='coerce').clip(-2,2)
    for c in ['date','target']:f[c]=pd.to_datetime(f[c])
    return f[['date','target','symbol','llm']]

def past_only_high(z):
    """Per symbol, compare today's novelty with the 80% quantile of strictly earlier novelty."""
    z=z.sort_values(['symbol','date']).copy()
    high=np.full(len(z),np.nan);prior=np.zeros(len(z),int)
    for _,g in z.groupby('symbol',sort=False):
        obs=[]
        for pos,(idx,row) in enumerate(g.iterrows()):
            loc=z.index.get_loc(idx)
            prior[loc]=len(obs)
            if np.isfinite(row.novelty):
                if len(obs)>=MIN_PRIOR_DAYS:
                    high[loc]=float(row.novelty>=np.quantile(obs,HIGH_Q))
                obs.append(float(row.novelty))
    z['prior_days']=prior
    z['high']=high
    return z

def tally(part,label):
    m=part[(part.label!=2)&part.llm.notna()&(part.llm!=0)]
    s=np.sign(m.ret_pct);hit=int((np.sign(m.llm)==s).sum())
    dis=m[(m.gap_z.notna())&(m.gap_z!=0)&(np.sign(m.llm)!=np.sign(m.gap_z))]
    return {'group':label,'rows':len(part),'moved_with_direction':len(m),'hits':hit,
            'hit_rate':hit/len(m) if len(m) else None,
            'p_one_sided':float(binomtest(hit,len(m),.5,alternative='greater').pvalue) if len(m) else None,
            'gap_hit_rate':float((np.sign(m.gap_z)==s).mean()) if len(m) else None,
            'disagree_n':len(dis),
            'disagree_hit_rate':float((np.sign(dis.llm)==np.sign(dis.ret_pct)).mean()) if len(dis) else None}

def main():
    z=pd.read_csv(OUT/'novelty_panel_coded.csv',parse_dates=['date','target'])
    f=SPEC['folds'][0]
    z=z[z.date<=pd.Timestamp(f['train'][1])].copy()
    z=z[~z.symbol.isin(SPEC['unseen_symbols'])].copy()
    assert z.symbol.nunique()==40 and z.target.max()<pd.Timestamp('2026-06-01')
    z=z.merge(llm_direction(),on=['date','target','symbol'],how='left',validate='one_to_one')
    z=past_only_high(z)
    z.to_csv(OUT/'t1_screen_panel.csv',index=False)
    rows=[tally(z[z.high==1],'high_novelty'),tally(z[z.high==0],'not_high_novelty'),
          tally(z[z.high.isna()&z.novelty.notna()],'held_insufficient_history'),
          tally(z[z.novelty.notna()],'all_with_novelty')]
    pd.DataFrame(rows).to_csv(OUT/'t1_screen.csv',index=False)
    h=rows[0]
    passed=bool(h['moved_with_direction']>=GATE['min_moved_with_direction']
                and h['disagree_n']>=GATE['min_disagree']
                and h['hit_rate'] is not None and h['hit_rate']>=GATE['hit_min']
                and h['p_one_sided']<GATE['one_sided_p_lt']
                and h['disagree_hit_rate'] is not None and h['disagree_hit_rate']>GATE['disagree_hit_gt'])
    failed=[k for k,ok in [('min_moved_with_direction',h['moved_with_direction']>=GATE['min_moved_with_direction']),
                           ('min_disagree',h['disagree_n']>=GATE['min_disagree']),
                           ('hit_min',bool(h['hit_rate'] and h['hit_rate']>=GATE['hit_min'])),
                           ('one_sided_p_lt',bool(h['p_one_sided'] is not None and h['p_one_sided']<GATE['one_sided_p_lt'])),
                           ('disagree_hit_gt',bool(h['disagree_hit_rate'] and h['disagree_hit_rate']>GATE['disagree_hit_gt']))] if not ok]
    res={'gate':GATE,'min_prior_days':MIN_PRIOR_DAYS,'high_quantile':HIGH_Q,
         'definition':'novelty threshold is the 80% quantile of the same symbol\'s strictly earlier novelty; '
                      'fewer than 20 prior observed days means the day is held out of the decision',
         'screen':rows,'passed':passed,'failed_criteria':failed,
         'four_fold_run':passed}
    (OUT/'t1_decision.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
    print(json.dumps(res,indent=2,default=str))

if __name__=='__main__':main()
