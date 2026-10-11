"""P1: gap-path and price-recency direction correction on top of B0. Development only, no API calls."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from src import score
from lab import oct10_models as M
from scripts import oct10_study as S

OUT=ROOT/'reports/followup_checks';OUT.mkdir(parents=True,exist_ok=True)
SPEC=json.loads((ROOT/'lab/folds.json').read_text(encoding='utf-8'))
GROUPS=json.loads((ROOT/'reports/oct10_extension/preregistered.json').read_text(encoding='utf-8'))['groups']
RAW=['gap_z','post_ret_z','pre_move_z','price_age_min','has_pre']
THRESH=0.65
B0_MEAN_REFERENCE=0.446866

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()[:16]

def load():
    d=pd.read_parquet(ROOT/'reports/oct10_extension/development.parquet')
    b=pd.read_parquet(ROOT/'cache/b_v1.parquet',columns=['date','symbol','post_ret','pre_move'])
    r=pd.read_parquet(ROOT/'cache/oct12_price_recency.parquet',columns=['date','target','symbol','price_age_min','has_pre'])
    d=d.merge(b,on=['date','symbol'],how='left',validate='one_to_one')
    d=d.merge(r,on=['date','target','symbol'],how='left',validate='one_to_one')
    assert d.target.max()<pd.Timestamp('2026-06-01')
    d['post_ret_z']=d.post_ret/d.vol20
    d['pre_move_z']=d.pre_move/d.vol20
    return d

def design(x,med):
    miss=(~np.isfinite(x)).astype(float)
    filled=np.where(np.isfinite(x),x,med)
    return np.column_stack([filled,miss])

def fit_direction(tr):
    """Logistic direction model on moved training rows. Impute and standardize on train only."""
    t=tr[tr.label!=2]
    x=t[RAW].to_numpy(float)
    med=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
    med=np.where(np.isfinite(med),med,0.)
    X=design(x,med)
    mean=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
    y=(t.ret_pct.to_numpy(float)>0).astype(int)
    model=LogisticRegression(C=1.,max_iter=2000)
    model.fit((X-mean)/sd,y)
    return {'median':med,'mean':mean,'sd':sd,'model':model}

def p_up(state,frame):
    X=design(frame[RAW].to_numpy(float),state['median'])
    return state['model'].predict_proba((X-state['mean'])/state['sd'])[:,1]

def correct(base_pred,prob,on):
    """Overwrite only rows where the model is confident and B0 points elsewhere."""
    p=base_pred.copy()
    if not on:return p,np.zeros(len(p),bool)
    up=(prob>=THRESH)&(base_pred<3)
    down=((1-prob)>=THRESH)&(base_pred>1)
    p=np.where(up,3,np.where(down,1,p)).astype(int)
    return p,(up|down)

def main():
    d=load()
    rows=[];preds=[];changes=[];switches=[]
    for f in SPEC['folds']:
        tr=d[d.date.between(*pd.to_datetime(f['train']))]
        va=d[d.date.between(*pd.to_datetime(f['val']))].sort_values(['date','symbol'])
        tr=tr[~tr.symbol.isin(SPEC['unseen_symbols'])]
        assert len(va)==3000
        base=M.fit_base(tr)
        pairs=[(a,b,M.fit_base(a)) for a,b in M.inner(tr)]
        gains=[]
        for a,b,p in pairs:
            st=fit_direction(a)
            bp=M.FrozenModel('B0',p).predict(M.inputs(b))
            prob=p_up(st,b)
            on_score=score(b.label,correct(bp,prob,True)[0])['score']
            off_score=score(b.label,correct(bp,prob,False)[0])['score']
            gains.append(on_score-off_score)
        on=bool(np.mean(gains)>0)
        state=fit_direction(tr)
        b0=M.FrozenModel('B0',base).predict(M.inputs(va))
        prob=p_up(state,va)
        p1,moved=correct(b0,prob,on)
        rows.append({'fold':f['name'],'inner_mean_gain':float(np.mean(gains)),
                     'inner_block_gains':[float(g) for g in gains],'correction_on':on,
                     'changed_rows':int(moved.sum()),
                     'confident_rows':int(((prob>=THRESH)|((1-prob)>=THRESH)).sum())})
        for kind,p in [('B0',b0),('P1',p1)]:
            q=va[['date','target','symbol','label']].copy()
            q['pred']=p;q['scenario']='train40';q['fold']=f['name'];q['kind']=kind
            preds.append(q)
        ch=va.loc[moved,['date','target','symbol','label','ret_pct','gap_z']].copy()
        ch['fold']=f['name'];ch['b0']=b0[moved];ch['p1']=p1[moved];ch['p_up']=prob[moved]
        changes.append(ch)
        if moved.any():
            b0_dir=np.sign(ch.b0-2);p1_dir=np.sign(ch.p1-2);true_dir=np.sign(ch.ret_pct)
            switches.append({'fold':f['name'],'changed':int(len(ch)),
                             'b0_wrong_direction':int((b0_dir==-true_dir).sum()),
                             'p1_wrong_direction':int((p1_dir==-true_dir).sum()),
                             'b0_flat_rows':int((ch.b0==2).sum()),
                             'fixed_opposite_errors':int(((b0_dir==-true_dir)&(p1_dir==true_dir)).sum()),
                             'new_opposite_errors':int(((b0_dir!=-true_dir)&(p1_dir==-true_dir)).sum())})
        print(f['name'],'on',on,'inner',round(float(np.mean(gains)),6),'changed',int(moved.sum()),flush=True)
    pred=pd.concat(preds,ignore_index=True)
    pred.to_parquet(OUT/'p1_predictions.parquet',index=False)
    pd.concat(changes,ignore_index=True).to_csv(OUT/'p1_changed_rows.csv',index=False)
    met=S.metrics(pred,GROUPS);met.to_csv(OUT/'p1_metrics.csv',index=False)
    ref=met[met.kind=='B0'].set_index(['fold','group']).score
    dd=met[met.kind=='P1'].set_index(['fold','group']).score-ref
    a=dd.xs('all',level='group')
    boot=S.bootstrap(pred,'P1')
    b0_all=ref.xs('all',level='group')
    dec={'kind':'P1','threshold':THRESH,'features':RAW,
         'B0_mean':float(b0_all.mean()),'B0_folds':b0_all.round(6).to_dict(),
         'B0_mean_reference':B0_MEAN_REFERENCE,'B0_mean_difference':float(b0_all.mean()-B0_MEAN_REFERENCE),
         'mean_gain':float(a.mean()),'fold_gain':a.round(6).to_dict(),'up_folds':int((a>0).sum()),
         'fold4_gain':float(a['fold4']),'unseen_gain':float(dd.xs('unseen',level='group').mean()),
         'balanced20_gain':float(dd.xs('balanced20',level='group').mean()),
         'seen_gain':float(dd.xs('seen',level='group').mean()),
         'bootstrap':boot,'lower_adjusted':boot['lo95'],
         'folds':rows,'direction_changes':switches}
    dec['passed']=bool(dec['mean_gain']>=.005 and dec['up_folds']>=3 and dec['fold4_gain']>=0
                       and dec['unseen_gain']>=0 and dec['balanced20_gain']>=0 and dec['lower_adjusted']>0)
    dec['input_digests']={p:digest(ROOT/p) for p in
                          ['reports/oct10_extension/development.parquet','cache/b_v1.parquet',
                           'cache/oct12_price_recency.parquet','cache/oct11_llm_full/full.parquet','lab/folds.json']}
    (OUT/'p1_decision.json').write_text(json.dumps(dec,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:v for k,v in dec.items() if k!='input_digests'},indent=2,default=str))

if __name__=='__main__':main()
