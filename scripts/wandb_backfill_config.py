"""Fill in the settings the first backfill pass left out.

scripts/wandb_backfill.py recorded each run's metrics, decision and source digests
but only wrote the protocol *path* for candidate runs, so the actual settings were
missing. Existing runs logged through lab.tracking all carry folds, folds_hash and
unseen_symbols too, and the backfilled ones did not, which makes them hard to
compare in the W&B UI.

This patches the configs of the runs already created; it does not make new runs and
does not touch any metric or summary value. folds_hash is computed the same way as
lab.tracking.folds_info, so it matches the 80057b71b4 the older runs carry.

    uv run python scripts/wandb_backfill_config.py --dry-run
    uv run python scripts/wandb_backfill_config.py
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.tracking import settings,has_key

SPEC=json.loads((ROOT/'lab/folds.json').read_text(encoding='utf-8'))

def read(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8'))

def folds_block():
    """Same shape and hash as lab.tracking.folds_info so old and new runs line up."""
    dates={f['name']:{'train':[f['train'][0],f['train'][1]],'val':[f['val'][0],f['val'][1]]}
           for f in SPEC['folds']}
    h=hashlib.sha1(json.dumps(dates,sort_keys=True).encode()).hexdigest()[:10]
    return {'folds':dates,'folds_hash':h,'unseen_symbols':SPEC['unseen_symbols'],
            'use_unseen':True,'method':'A-folds4 (주 지표 fold4)',
            'holdout_from':SPEC['holdout_from'],
            # named score_fn, not score: summary.score already holds the number
            'score_fn':'src.data.score'}

# The bar every family was judged against, written before each run.
ADOPTION={'mean_gain':'>= 0.005','up_folds':'>= 3','fold4_gain':'>= 0','unseen_gain':'>= 0',
          'balanced20_gain':'>= 0',
          'bootstrap':'5-trading-day blocks of 50-symbol days, 2000 draws, seed 2026, '
                      'family-adjusted lower quantile > 0',
          'source':'docs/worklog 2026-10-10 onward; identical across the families below'}

def family_settings():
    """The real settings per family, read from the committed protocol files."""
    pre=read('reports/oct10_extension/preregistered.json')
    fup=read('reports/followup_checks/config.json')
    cost=read('reports/followup_checks/cost_decode_decision.json')
    return {
      'final_study':{'protocol_file':'reports/final_study/frozen.json',
                     'frozen':read('reports/final_study/frozen.json'),
                     'fit_parameters':read('reports/final_study/fit_parameters.json'),
                     'oct10_protocol':read('lab/oct10_protocol.json'),
                     'groups':pre['groups'],'adoption_criteria':ADOPTION},
      'oct10_extension':{'protocol_file':'reports/oct10_extension/preregistered.json',
                     'preregistered':{k:v for k,v in pre.items() if k!='groups'},
                     'groups':pre['groups'],
                     'oct10_protocol':read('lab/oct10_protocol.json'),
                     'adoption_criteria':ADOPTION},
      'followup_p1':{'protocol_file':'reports/followup_checks/config.json',
                     'p1':fup['p1'],'scope':fup['scope'],'order':fup['order'],'seed':fup['seed'],
                     'stop_reasons':fup['stop_reasons'],'groups':pre['groups'],
                     'adoption_criteria':ADOPTION},
      'followup_cost_decode':{'protocol_file':'reports/followup_checks/cost_decode_decision.json',
                     'taus':cost['taus'],'tau_chosen':cost['tau_chosen'],
                     'decoders':{'aggr':'train label quantiles on the expected class (the earlier way)',
                                 'cost':'per row argmin of sum_y p(y) W[y,k]; tau = 0',
                                 'ratio':'per row argmin of cost_k - tau * base_k, base from the train marginal'},
                     'tie_break':'ties go to flat; W row 2 is all zeros so a bare argmin would answer 0',
                     'lgb_params':'work/a/lgb.py PARAMS, B features 26, early stop on the last 40 train days',
                     'class_weights_for_wlgb':{'0':4.0,'1':1.0,'2':0.25,'3':1.0,'4':4.0},
                     'groups':pre['groups'],'adoption_criteria':ADOPTION},
      # Audits produce no candidate score, so they carry their own protocol only.
      'lecture_extension':{},'oct11_diagnostics':{},'oct11_llm_direction':{},
      'oct12_lgb_gap':{'protocol_file':'reports/oct12_checks/protocol.json',
                       'protocol':read('reports/oct12_checks/protocol.json')},
      'oct12_collection':{'protocol_file':'reports/oct12_checks/protocol.json',
                       'protocol':read('reports/oct12_checks/protocol.json')['collection']},
      'oct12_novelty':{'protocol_file':'reports/oct12_checks/protocol.json',
                       'protocol':read('reports/oct12_checks/protocol.json')['novelty']},
      'followup_recount':{'protocol_file':'reports/followup_checks/config.json',
                       'order':read('reports/followup_checks/config.json')['order']},
      'followup_t1':{'protocol_file':'reports/followup_checks/config.json',
                       't1':read('reports/followup_checks/config.json')['t1'],
                       'adoption_criteria':ADOPTION},
      'followup_metric_robustness':{'protocol_file':'reports/followup_checks/metric_robustness_protocol.json',
                       'protocol':read('reports/followup_checks/metric_robustness_protocol.json')},
      'final_study_holdout':{'protocol_file':'reports/final_study/frozen.json',
                       'frozen':read('reports/final_study/frozen.json'),
                       'holdout_rule':'opened once on 2026-10-09 after the winner was frozen; '
                                      'never reused for selection'},
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    s=settings();base=f"{s['entity']}/{s['project']}"
    if not has_key():
        print('no W&B key; nothing changed');return
    import wandb
    api=wandb.Api()
    fams=family_settings();shared=folds_block()
    runs=list(api.runs(base,filters={'tags':'backfill'}))
    print(f'{len(runs)} backfilled runs in {base}')
    missing=sorted({r.config.get('family') for r in runs}-set(fams))
    assert not missing,f'no settings defined for {missing}'
    for r in sorted(runs,key=lambda r:r.name):
        fam=r.config.get('family')
        add={k:v for k,v in {**shared,**fams[fam]}.items() if k not in r.config}
        if a.dry_run:
            print(f'  {r.name}: +{sorted(add)}' if add else f'  {r.name}: already complete')
            continue
        if add:
            r.config.update(add);r.update()
        print(f'  {r.name}: +{len(add)} keys')
    if a.dry_run:print('\ndry run: nothing was changed')
    else:print('\ndone')

if __name__=='__main__':main()
