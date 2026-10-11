"""Step 1: separate missing codes in the earlier oct12 tallies. Reports before and after."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import binomtest

OUT=ROOT/'reports/followup_checks';OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'reports/oct12_checks'
SPEC=json.loads((ROOT/'lab/folds.json').read_text(encoding='utf-8'))
TRAIN1_END=pd.Timestamp(SPEC['folds'][0]['train'][1])

def novelty_codes():
    z=pd.read_csv(SRC/'novelty_panel.csv',parse_dates=['date','target'])
    in_scope=z.date<=TRAIN1_END
    code=pd.Series('not_computed_out_of_screen_scope',index=z.index)
    code[in_scope&(z.news_count==0)]='no_articles'
    code[in_scope&(z.news_count>0)&z.novelty.isna()]='no_prior_comparison'
    code[in_scope&z.novelty.notna()&(z.novelty>0)]='computed_positive'
    code[in_scope&z.novelty.notna()&(z.novelty==0)]='computed_exact_zero'
    z['novelty_code']=code
    gap=pd.Series('usable',index=z.index)
    gap[z.gap_z.isna()]='gap_missing'
    gap[z.gap_z.fillna(1)==0]='gap_exact_zero'
    z['gap_code']=gap
    z.to_csv(OUT/'novelty_panel_coded.csv',index=False)
    counts=z.novelty_code.value_counts().to_dict()
    gapcounts=z[in_scope].gap_code.value_counts().to_dict()
    # Rescreen the fold1-train tone direction with gap-missing and gap-zero rows removed.
    train=z[in_scope].copy()
    threshold=float(train.novelty.quantile(0.8))
    train['high']=train.novelty>=threshold
    rows=[]
    for name,part in [('all_news',train[train.news_count>0]),('high_novelty',train[train.high]),
                      ('other_news',train[(train.news_count>0)&~train.high])]:
        m=part[(part.label!=2)&(part.direction!=0)]
        s=np.sign(m.ret_pct)
        hit=int((m.direction==s).sum())
        old=m[m.direction!=np.sign(m.gap_z)]
        new=m[(m.gap_code=='usable')&(m.direction!=np.sign(m.gap_z))]
        rows.append({'group':name,'rows':len(part),'moved_nonzero':len(m),'hits':hit,
                     'hit_rate':hit/len(m) if len(m) else None,
                     'p_one_sided':float(binomtest(hit,len(m),.5,alternative='greater').pvalue) if len(m) else None,
                     'disagree_n_before':len(old),
                     'disagree_hit_rate_before':float((old.direction==np.sign(old.ret_pct)).mean()) if len(old) else None,
                     'disagree_n_after':len(new),
                     'disagree_hit_rate_after':float((new.direction==np.sign(new.ret_pct)).mean()) if len(new) else None,
                     'dropped_gap_missing_or_zero':len(old)-len(new)})
    pd.DataFrame(rows).to_csv(OUT/'novelty_screen_recounted.csv',index=False)
    return {'novelty_code_counts':counts,'gap_code_counts_in_scope':gapcounts,
            'fold1_train_threshold':threshold,'rescreen':rows,
            'unit_note':'novelty = 1 - max TF-IDF cosine to the previous 20 symbol-day documents; '
                        'the protocol fixed this expression, and the run kept it. Values are a '
                        'similarity complement in [0,1], not a count.'}

def text_stages():
    out={}
    for name in ['news','reddit']:
        s=pd.read_csv(SRC/f'{name}_collection_summary.csv')
        a=s[s.group=='all'].iloc[0].to_dict()
        requested=float(a['selected']);answered=float(a['llm_result']);validated=float(a['validated'])
        out[name]={'requested_to_api':requested,'answered_by_api':answered,
                   'validated':validated,
                   'no_response_or_unprocessed':requested-answered,
                   'answered_but_failed_validation':answered-validated,
                   'note':'answered counts rows the API actually returned (valid + excluded files). '
                          'Requests with no stored row are unprocessed or errored and are not counted as responses.'}
        if name=='news':
            out[name]['raw_symbol_links']=float(a['raw'])
            out[name]['after_expectation_filter']=float(a['expectation_filter'])
            out[name]['pre_filter_candidate']='unavailable: the handoff stores only the post-filter count'
    # Company relevance is the LLM's own judgement, available for the full extraction.
    f=pd.read_parquet(ROOT/'cache/oct11_llm_full/full.parquet',columns=['date','target','symbol','raw'])
    j=f.raw.apply(lambda s:json.loads(s) if isinstance(s,str) else {})
    rel=pd.to_numeric(j.str.get('relevant').map({True:1,False:0}),errors='coerce')
    out['news_relevance']={'symbol_days_sent':int(len(f)),
                           'symbol_days_judged_relevant':int(rel.fillna(0).sum()),
                           'relevant_share':float(rel.fillna(0).mean()),
                           'note':'a symbol-day linked by the raw feed is not necessarily company-relevant, '
                                  'so a drop from the raw link count is not by itself a lost useful article.'}
    return out

def main():
    before=json.loads((SRC/'novelty_decision.json').read_text(encoding='utf-8'))
    result={'before':{'novelty_screen':before['screen'],'novelty_passed':before['passed']},
            'after':{'novelty':novelty_codes(),'text_stages':text_stages()}}
    (OUT/'step1_recount.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result['after'],indent=2,default=str))

if __name__=='__main__':main()
