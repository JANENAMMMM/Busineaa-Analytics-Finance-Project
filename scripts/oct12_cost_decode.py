"""Expected-cost decoding of the multiclass LightGBM probabilities.

Earlier loss work only changed the training objective (L2, Huber, L1, ordinal,
score-weighted multiclass) and always decoded by matching train label quantiles.
This closes the remaining gap: keep the probabilities and decode with the
official score weights themselves.

For a row with class probabilities p, predicting k costs cost_k = sum_y p(y) W[y,k].
The official score is 1 - N/D, and predicting k also raises the chance denominator
by base_k = sum_y q(y) W[y,k] for the train label marginal q. With the current
ratio r = N/D, the locally best choice minimises cost_k - r * base_k, so pure cost
minimisation (tau = 0) is the conservative special case.
"""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src import score
from src.data import WEIGHT
from lab import oct10_models as M
from work.a import lgb as L
from work.a.models2 import CLASS_W
from scripts import oct10_study as S

OUT=ROOT/'reports/followup_checks';OUT.mkdir(parents=True,exist_ok=True)
SPEC=json.loads((ROOT/'lab/folds.json').read_text(encoding='utf-8'))
GROUPS=json.loads((ROOT/'reports/oct10_extension/preregistered.json').read_text(encoding='utf-8'))['groups']
TAUS=[0.0,0.3,0.45,0.55,0.7]
B0_MEAN_REFERENCE=0.446866
LGB_AGGR_REFERENCE={'fold1':0.346125,'fold2':0.420329,'fold3':0.391994,'fold4':0.495440}

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()[:16]

def load():
    d=pd.read_parquet(ROOT/'reports/oct10_extension/development.parquet')
    t=pd.read_parquet(ROOT/'cache/b_v1.parquet')
    assert d.target.max()<pd.Timestamp('2026-06-01') and t.date.max()<pd.Timestamp('2026-06-01')
    # The two tables share gap_z, ext_range_z, vol20, ret1_z, ret5_z, earn_now, label and
    # ret_pct and agree to float noise; keep the development copies so B0 stays bit-exact.
    shared=[c for c in d.columns if c in t.columns and c not in ('date','symbol')]
    t=t.drop(columns=shared)
    z=d.merge(t,on=['date','symbol'],how='left',validate='one_to_one')
    assert len(z)==len(d)
    return z

def base_cost(labels):
    """Chance-denominator contribution of each predicted class under the train marginal."""
    q=np.bincount(np.asarray(labels,int),minlength=5)/len(labels)
    return (q[:,None]*WEIGHT).sum(0)

TIE_BREAK=np.abs(np.arange(5)-2)*1e-9

def decode(proba,base,tau):
    """Ties break towards flat: a row the model is sure is flat costs the same whatever we
    say (W row 2 is all zeros), and a bare argmin would answer 0 for every such row."""
    return np.argmin(proba@WEIGHT-tau*base+TIE_BREAK,axis=1).astype(int)

def train_probs(tr,va,weighted):
    """Same fold-wise fit as the earlier audit: early stop on the last 40 train days, refit on all."""
    import lightgbm as lgb
    feats=L.FEATS
    params=dict(L.PARAMS)
    w=(lambda frame:frame.label.map(CLASS_W)) if weighted else (lambda frame:None)
    ds=lambda frame:lgb.Dataset(frame[feats],frame.label,weight=w(frame))
    cut=np.sort(tr.date.unique())[-L.VALID_DAYS]
    fit,inner=tr[tr.date<cut],tr[tr.date>=cut]
    es=lgb.train(params,ds(fit),2000,valid_sets=[ds(inner)],
                 callbacks=[lgb.early_stopping(100,verbose=False)])
    rounds=max(es.best_iteration,20)
    booster=lgb.train(params,ds(tr),rounds)
    return booster.predict(va[feats]),booster.predict(tr[feats]),inner,es,rounds

def aggr_labels(tr,va,es,inner,booster_tr,booster_va):
    """The decoding the earlier audit used, kept as the within-model reference."""
    q=L.props(tr.label)
    e_es_tr=es.predict(tr[tr.date<inner.date.min()][L.FEATS])@L.LV
    e_inner=es.predict(inner[L.FEATS])@L.LV
    ks={k:score(inner.label,L.cut(e_inner,L.thresholds(e_es_tr,L.aggr_q(q,k))))['score'] for k in L.KS}
    k=max(ks,key=ks.get)
    th=L.thresholds(booster_tr@L.LV,L.aggr_q(q,k))
    return L.cut(booster_va@L.LV,th),k

def main():
    z=load()
    preds=[];folds=[];tau_choice={}
    for f in SPEC['folds']:
        tr=z[z.date.between(*pd.to_datetime(f['train']))&~z.symbol.isin(SPEC['unseen_symbols'])].copy()
        va=z[z.date.between(*pd.to_datetime(f['val']))].sort_values(['date','symbol']).copy()
        assert len(va)==3000 and va.date.nunique()==60
        tr['label']=tr.label.astype(int)
        base=base_cost(tr.label)
        b0=M.FrozenModel('B0',M.fit_base(tr)).predict(M.inputs(va))
        row={'fold':f['name'],'train_marginal_base_cost':base.round(4).tolist()}
        for weighted,tag in [(False,'lgb'),(True,'wlgb')]:
            p_va,p_tr,inner,es,rounds=train_probs(tr,va,weighted)
            agg,k=aggr_labels(tr,va,es,inner,p_tr,p_va)
            # tau is picked on the existing inner 3 blocks with a 12-day embargo, never on val.
            gains={t:[] for t in TAUS}
            for a,b in M.inner(tr):
                a=a.copy();a['label']=a.label.astype(int)
                pb,_,_,_,_=train_probs(a,b,weighted)
                ib=base_cost(a.label)
                for t in TAUS:gains[t].append(score(b.label,decode(pb,ib,t))['score'])
            tau=max(TAUS,key=lambda t:(float(np.mean(gains[t])),-t))
            tau_choice.setdefault(f['name'],{})[tag]=tau
            row[f'{tag}_rounds']=rounds;row[f'{tag}_aggr_k']=k;row[f'{tag}_tau']=tau
            row[f'{tag}_inner_tau_scores']={str(t):float(np.mean(v)) for t,v in gains.items()}
            for kind,pred in [(f'{tag}_aggr',agg),(f'{tag}_cost',decode(p_va,base,0.)),
                              (f'{tag}_ratio',decode(p_va,base,tau))]:
                q=va[['date','target','symbol','label']].copy()
                q['pred']=pred;q['scenario']='train40';q['fold']=f['name'];q['kind']=kind
                preds.append(q)
        q=va[['date','target','symbol','label']].copy()
        q['pred']=b0;q['scenario']='train40';q['fold']=f['name'];q['kind']='B0'
        preds.append(q)
        folds.append(row)
        print(f['name'],'tau',tau_choice[f['name']],flush=True)
    pred=pd.concat(preds,ignore_index=True)
    pred.to_parquet(OUT/'cost_decode_predictions.parquet',index=False)
    met=S.metrics(pred,GROUPS);met.to_csv(OUT/'cost_decode_metrics.csv',index=False)
    ref=met[met.kind=='B0'].set_index(['fold','group']).score
    b0_all=ref.xs('all',level='group')
    rows=[]
    for kind in sorted(set(pred.kind)-{'B0'}):
        dd=met[met.kind==kind].set_index(['fold','group']).score-ref
        a=dd.xs('all',level='group')
        boot=S.bootstrap(pred,kind)
        r={'kind':kind,'fold_score':met[(met.kind==kind)&(met.group=='all')].set_index('fold').score.round(6).to_dict(),
           'mean_score':float(met[(met.kind==kind)&(met.group=='all')].score.mean()),
           'mean_gain_vs_B0':float(a.mean()),'fold_gain':a.round(6).to_dict(),'up_folds':int((a>0).sum()),
           'fold4_gain':float(a['fold4']),'unseen_gain':float(dd.xs('unseen',level='group').mean()),
           'balanced20_gain':float(dd.xs('balanced20',level='group').mean()),
           'lower_adjusted':boot['lo95'],'bootstrap':boot,
           'pred_distribution':(pred[pred.kind==kind].pred.value_counts(normalize=True)
                                .reindex(range(5)).fillna(0).round(4).to_dict())}
        r['passed']=bool(r['mean_gain_vs_B0']>=.005 and r['up_folds']>=3 and r['fold4_gain']>=0
                         and r['unseen_gain']>=0 and r['balanced20_gain']>=0 and r['lower_adjusted']>0)
        rows.append(r)
    res={'taus':TAUS,'tau_chosen':tau_choice,
         'B0_mean':float(b0_all.mean()),'B0_folds':b0_all.round(6).to_dict(),
         'B0_mean_reference':B0_MEAN_REFERENCE,'B0_mean_difference':float(b0_all.mean()-B0_MEAN_REFERENCE),
         'lgb_aggr_reference':LGB_AGGR_REFERENCE,'folds':folds,'candidates':rows,
         'label_marginal_note':'base_k uses the train label marginal of the 40 training symbols only',
         'limitation':'repeatedly inspected development data; no holdout labels or scores were read'}
    (OUT/'cost_decode_decision.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
    res['input_digests']={p:digest(ROOT/p) for p in
                          ['reports/oct10_extension/development.parquet','cache/b_v1.parquet','lab/folds.json']}
    (OUT/'cost_decode_decision.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
    print(met[met.group=='all'].pivot(index='kind',columns='fold',values='score')
          .assign(mean=lambda x:x.mean(axis=1)).round(6))
    print(json.dumps([{k:r[k] for k in ['kind','mean_score','mean_gain_vs_B0','up_folds','fold4_gain','lower_adjusted','passed']}
                      for r in rows],indent=2))

if __name__=='__main__':main()
