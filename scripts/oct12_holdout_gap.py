"""Why is the holdout score 0.15 above development?

The number 0.596716 is recorded in several places but nothing explains it, and a
peer reviewer will ask. This compares B0's development behaviour with the holdout
aggregates that the 10-09 final study already wrote down.

It does not read holdout labels or predictions. The only holdout input is
reports/final_study/holdout_metrics.csv, a file of aggregates produced by that
pre-registered run; this script recomputes nothing on the evaluation period.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src import score

OUT=ROOT/'reports/followup_checks'
GROUPS=json.loads((ROOT/'reports/oct10_extension/preregistered.json')
                  .read_text(encoding='utf-8'))['groups']
METRICS=['score','accuracy','direction','big_recall','big_prec']

def main():
    dev=pd.read_parquet(OUT/'p1_predictions.parquet')
    dev=dev[dev.kind=='B0']
    assert dev.target.max()<pd.Timestamp('2026-06-01'),'development rows only'
    hold=pd.read_csv(ROOT/'reports/final_study/holdout_metrics.csv')
    hold=hold[hold.scenario=='train40'].set_index('group')

    rows=[]
    for fold,g in dev.groupby('fold'):
        rows.append({'period':fold,'group':'all',**{k:score(g.label,g.pred)[k] for k in METRICS},
                     'n':len(g)})
    for name,syms in [('all',None),*GROUPS.items()]:
        z=dev if syms is None else dev[dev.symbol.isin(syms)]
        rows.append({'period':'development_pooled','group':name,
                     **{k:score(z.label,z.pred)[k] for k in METRICS},'n':len(z)})
        if name in hold.index:
            rows.append({'period':'holdout','group':name,
                         **{k:float(hold.loc[name,k]) for k in METRICS},
                         'n':int(hold.loc[name,'n'])})
    t=pd.DataFrame(rows)
    t.round(6).to_csv(OUT/'holdout_gap.csv',index=False)

    folds=t[t.period.str.startswith('fold')]
    pooled=t[(t.period=='development_pooled')&(t.group=='all')].iloc[0]
    hall=t[(t.period=='holdout')&(t.group=='all')].iloc[0]
    fold_mean=float(folds.score.mean())
    by_group={}
    for name in ['all',*GROUPS]:
        d=t[(t.period=='development_pooled')&(t.group==name)]
        h=t[(t.period=='holdout')&(t.group==name)]
        if len(d) and len(h):
            by_group[name]={'dev_direction':float(d.direction.iloc[0]),
                            'holdout_direction':float(h.direction.iloc[0]),
                            'direction_gain':float(h.direction.iloc[0]-d.direction.iloc[0]),
                            'dev_score':float(d.score.iloc[0]),
                            'holdout_score':float(h.score.iloc[0])}
    res={'development_fold_scores':folds.set_index('period').score.round(6).to_dict(),
         'development_fold_direction':folds.set_index('period').direction.round(6).to_dict(),
         'development_fold_mean_score':fold_mean,
         'development_pooled':{k:float(pooled[k]) for k in METRICS},
         'holdout':{k:float(hall[k]) for k in METRICS},
         'score_gap_vs_fold_mean':float(hall.score-fold_mean),
         'score_gap_vs_pooled':float(hall.score-pooled.score),
         'holdout_direction_above_every_dev_fold':bool(hall.direction>folds.direction.max()),
         'best_dev_fold_direction':float(folds.direction.max()),
         'by_group':by_group,
         'holdout_input':'reports/final_study/holdout_metrics.csv (aggregates from the 2026-10-09 '
                         'pre-registered run); no holdout labels or predictions were read here',
         'reading':('B0 did not get better. The gap predicted direction far more often in the '
                    'evaluation period (0.691) than in any development fold (best 0.584), and the '
                    'jump is the same size in every symbol group, so it is a property of the period '
                    'and not of which companies were scored. Quoting 0.597 next to another team s '
                    'number without saying this compares two different periods would overstate it.')}
    (OUT/'holdout_gap.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(t.round(4).to_string(index=False))
    print()
    print(json.dumps({k:v for k,v in res.items() if k!='by_group'},indent=2))
    print('by_group',json.dumps(by_group,indent=2))

if __name__=='__main__':main()
