"""Backfill the 10-09 onward experiments into W&B.

Those runs were never logged because the scripts from 10-09 on drive their own
fold loops instead of lab.cv.run_cv, so lab.tracking is never called. Everything
here is read back from the committed reports/ artefacts; nothing is recomputed and
no model is refit, so a backfilled run says exactly what its report file says.

W&B stamps a run with the time it was created, which would make all of these look
like 10-12 work. Every run therefore carries the real experiment date in its name,
in the tag orig:<date> and in config.original_date, and the tag "backfill".

Row-level predictions and raw data are not uploaded, matching lab/tracking.py.

    uv run python scripts/wandb_backfill.py --dry-run   # print the plan, upload nothing
    uv run python scripts/wandb_backfill.py             # create the runs
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from lab.tracking import settings,git_info,has_key

METRICS=['score','accuracy','direction','big_recall','big_prec','n']

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()[:16]

def read_json(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8'))

# ---------------------------------------------------------------- inventory

def model_runs():
    """One run per candidate, built from the shared metrics schema."""
    out=[]
    fams=[
      dict(date='2026-10-09',family='final_study',metrics='reports/final_study/dev_metrics.csv',
           protocol='reports/final_study/frozen.json',
           decisions={d['kind']:d for d in read_json('reports/final_study/selection.json')['decisions']},
           note='frozen post-selection comparison; B0 won and was carried to the holdout',
           extra={'winner':read_json('reports/final_study/selection.json')['winner'],
                  'api_spend_krw':read_json('reports/final_study/selection.json').get('api_spend_krw')}),
      dict(date='2026-10-10',family='oct10_extension',metrics='reports/oct10_extension/metrics.csv',
           protocol='reports/oct10_extension/preregistered.json',
           decisions={d['kind']:d for d in read_json('reports/oct10_extension/decisions.json')['decisions']},
           note='text activity and per-symbol boundary candidates; none adopted',extra={}),
      dict(date='2026-10-12',family='followup_p1',metrics='reports/followup_checks/p1_metrics.csv',
           protocol='reports/followup_checks/config.json',
           decisions={'P1':read_json('reports/followup_checks/p1_decision.json')},
           note='gap path and price recency direction correction; not adopted',extra={}),
      dict(date='2026-10-12',family='followup_cost_decode',metrics='reports/followup_checks/cost_decode_metrics.csv',
           protocol='reports/followup_checks/cost_decode_decision.json',
           decisions={c['kind']:c for c in read_json('reports/followup_checks/cost_decode_decision.json')['candidates']},
           note='expected-cost decoding of the multiclass LightGBM probabilities; none adopted',extra={}),
    ]
    for fam in fams:
        m=pd.read_csv(ROOT/fam['metrics'])
        for kind in sorted(m.kind.unique()):
            # B0 is the reference in every family; log it once, from the final study.
            if kind=='B0' and fam['family']!='final_study':continue
            out.append(dict(kind=kind,**fam))
    return out

def audit_runs():
    """Checks that produce no candidate score: permutations, ALE, coverage, screens."""
    return [
      dict(date='2026-10-09',family='lecture_extension',name='audit_lecture_extension',
           note='block permutation, ALE, exact SHAP, lagged-score and text quality checks on B0',
           files=['reports/lecture_extension/summary.json','reports/lecture_extension/manifest.json',
                  'reports/lecture_extension/text_quality.json'],
           tables={'class_metrics':'reports/lecture_extension/class_metrics.csv',
                   'block_importance':'reports/lecture_extension/block_importance.csv',
                   'ale':'reports/lecture_extension/ale.csv',
                   'news_measurement':'reports/lecture_extension/news_measurement.csv',
                   'reddit_measurement':'reports/lecture_extension/reddit_measurement.csv'}),
      dict(date='2026-10-11',family='oct11_diagnostics',name='audit_e1_e3_e4',
           note='where the weighted loss goes (e1), per volatility-bucket boundaries (e3), oracle headroom (e4)',
           files=[],
           tables={'group_fold':'reports/oct11_e1/group_fold.csv','symbol':'reports/oct11_e1/symbol.csv',
                   'vol_quintile':'reports/oct11_e1/vol_quintile.csv',
                   'e3_volbucket':'reports/oct11_e1/e3_volbucket.csv',
                   'e4_headroom':'reports/oct11_e1/e4_headroom.csv'}),
      dict(date='2026-10-11',family='oct11_llm_direction',name='screen_llm_direction',
           note='gpt-4.1-mini direction over all 18,802 symbol-days; failed the fold1 screen so no four-fold run',
           files=['reports/oct11_llm/result.json'],tables={}),
      dict(date='2026-10-12',family='oct12_lgb_gap',name='audit_lgb_gap_permutation_ale',
           note='joint permutation of gap, gap_z and gap_rel_z plus ALE on the LightGBM model',
           files=['reports/oct12_checks/lgb_gap_fit.json','reports/oct12_checks/protocol.json'],
           tables={'permutations':'reports/oct12_checks/lgb_gap_permutations.csv',
                   'ale':'reports/oct12_checks/lgb_gap_ale.csv'}),
      dict(date='2026-10-12',family='oct12_collection',name='audit_text_collection',
           note='stage by stage news and Reddit coverage; labels never read',
           files=[],tables={'news':'reports/oct12_checks/news_collection_summary.csv',
                            'reddit':'reports/oct12_checks/reddit_collection_summary.csv',
                            'reddit_raw_by_day':'reports/oct12_checks/reddit_raw_by_day.csv'}),
      dict(date='2026-10-12',family='oct12_novelty',name='screen_title_novelty',
           note='past-only TF-IDF title novelty with news tone direction; failed the fold1 screen',
           files=['reports/oct12_checks/novelty_decision.json'],
           tables={'screen':'reports/oct12_checks/novelty_screen.csv'}),
      dict(date='2026-10-12',family='followup_recount',name='audit_tally_recount',
           note='separates novelty missing codes and API request, response and validation counts',
           files=['reports/followup_checks/step1_recount.json'],
           tables={'novelty_screen_recounted':'reports/followup_checks/novelty_screen_recounted.csv'}),
      dict(date='2026-10-12',family='followup_t1',name='screen_novelty_llm_direction',
           note='existing LLM direction on high-novelty days; failed the fold1 screen, candidate terminated',
           files=['reports/followup_checks/t1_decision.json'],
           tables={'screen':'reports/followup_checks/t1_screen.csv'}),
      dict(date='2026-10-12',family='followup_metric_robustness',name='audit_metric_robustness',
           note='re-scores the 18 frozen candidates under the official weights and four variants',
           files=['reports/followup_checks/metric_robustness.json',
                  'reports/followup_checks/metric_robustness_protocol.json'],
           tables={'robustness':'reports/followup_checks/metric_robustness.csv'}),
    ]

def holdout_run():
    return dict(date='2026-10-09',family='final_study_holdout',name='holdout_B0',
                note='the one holdout pass, opened 2026-10-09 after the winner was frozen',
                files=['reports/final_study/HOLDOUT_OPENED.json','reports/final_study/holdout_done.json',
                       'reports/final_study/clean_reproduction.json','reports/final_study/clean_grader_check.json'],
                tables={'holdout_metrics':'reports/final_study/holdout_metrics.csv'})

# ---------------------------------------------------------------- logging

def base_config(date,family,files,note):
    cfg={'original_date':date,'family':family,'backfilled':True,
         'backfill_note':'logged from committed reports/ artefacts; nothing recomputed',
         'description':note,'git':git_info(),
         'source_files':{f:digest(ROOT/f) for f in files if (ROOT/f).exists()}}
    return cfg

def log_model_run(wandb,spec,dry):
    m=pd.read_csv(ROOT/spec['metrics'])
    z=m[m.kind==spec['kind']]
    primary=z[(z.scenario=='train40')&(z.group=='all')].sort_values('fold')
    mean=float(primary.score.mean())
    name=f"{spec['date']}_{spec['family']}_{spec['kind']}"
    dec=spec['decisions'].get(spec['kind'],{})
    adopted=bool(dec.get('adoptable') or dec.get('passed') or dec.get('development_gate_passed'))
    # B0 is the submitted model, not a challenger, so selection.json holds no decision row for
    # it. Without this it would fall through to "not_adopted", which reads as the opposite.
    submitted=spec['kind']=='B0'
    tags=['backfill',f"orig:{spec['date']}",spec['family'],
          'submitted' if submitted else 'adopted' if adopted else 'not_adopted']
    cfg=base_config(spec['date'],spec['family'],[spec['metrics'],spec['protocol']],spec['note'])
    cfg['kind']=spec['kind'];cfg['protocol_file']=spec['protocol'];cfg.update(spec['extra'] or {})
    summary={'score':mean,'score_main':float(primary.score.iloc[-1]) if len(primary) else None,
             'n_folds':int(len(primary))}
    for _,r in z.iterrows():
        for k in METRICS:
            if pd.notna(r[k]):summary[f"{r.scenario}/{r.fold}/{r.group}/{k}"]=float(r[k])
    for k,v in (dec or {}).items():
        if isinstance(v,(int,float,str,bool)) or v is None:summary[f"decision/{k}"]=v
    summary['decision/adopted']=adopted
    summary['decision/submitted_model']=submitted
    if dry:
        print(f"  {name}  score={mean:.6f} tags={tags} summary_keys={len(summary)}")
        return
    run=wandb.init(entity=S['entity'],project=S['project'],name=name,config=cfg,tags=tags,
                   job_type='cv_backfill',dir=str(ROOT),reinit='finish_previous')
    run.log({'cv_table':wandb.Table(dataframe=z.round(6))})
    for k,v in summary.items():run.summary[k]=v
    run.finish()

def log_audit_run(wandb,spec,dry,job='audit_backfill'):
    tags=['backfill',f"orig:{spec['date']}",spec['family'],
          'holdout' if spec['family']=='final_study_holdout' else 'audit']
    cfg=base_config(spec['date'],spec['family'],list(spec['files'])+list(spec['tables'].values()),spec['note'])
    for f in spec['files']:
        if (ROOT/f).exists():cfg[Path(f).stem]=read_json(f)
    if dry:
        print(f"  {spec['name']}  tags={tags} json={len(spec['files'])} tables={list(spec['tables'])}")
        return
    run=wandb.init(entity=S['entity'],project=S['project'],name=f"{spec['date']}_{spec['name']}",
                   config=cfg,tags=tags,job_type=job,dir=str(ROOT),reinit='finish_previous')
    for key,rel in spec['tables'].items():
        if (ROOT/rel).exists():
            run.log({key:wandb.Table(dataframe=pd.read_csv(ROOT/rel).round(6))})
    if spec['family']=='final_study_holdout':
        h=pd.read_csv(ROOT/'reports/final_study/holdout_metrics.csv')
        for _,r in h.iterrows():
            for k in METRICS:
                if pd.notna(r[k]):run.summary[f"holdout/{r.scenario}/{r.group}/{k}"]=float(r[k])
        run.summary['score']=float(h[(h.scenario=='train40')&(h.group=='all')].score.iloc[0])
    run.finish()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    global S;S=settings()
    models=model_runs();audits=audit_runs();hold=holdout_run()
    print(f"target: {S['entity']}/{S['project']}")
    print(f"plan: {len(models)} candidate runs + {len(audits)} audit runs + 1 holdout run "
          f"= {len(models)+len(audits)+1} total")
    print('\ncandidate runs')
    wandb=None
    if not a.dry_run:
        if not has_key():
            print('no W&B key; nothing uploaded');return
        import wandb as wandb
    for s in models:log_model_run(wandb,s,a.dry_run)
    print('\naudit runs')
    for s in audits:log_audit_run(wandb,s,a.dry_run)
    print('\nholdout run')
    log_audit_run(wandb,hold,a.dry_run,job='holdout_backfill')
    if a.dry_run:print('\ndry run: nothing was uploaded')
    else:print('\ndone')

if __name__=='__main__':main()
