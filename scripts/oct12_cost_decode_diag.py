"""Why each decoder scored what it did: precision of the extreme calls, direction errors, accuracy.

The official score divides by a chance denominator that grows when a prediction is
surprising, so a decoder can raise the score while getting fewer rows right. These
columns separate that from real skill.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd

OUT=ROOT/'reports/followup_checks'

def main():
    p=pd.read_parquet(OUT/'cost_decode_predictions.parquet')
    res=json.loads((OUT/'cost_decode_decision.json').read_text(encoding='utf-8'))
    label=p[p.kind=='B0'].label.to_numpy(int)
    rows=[]
    for kind,z in p.groupby('kind'):
        moved=z[z.label!=2]
        d=np.sign(moved.pred-2)*np.sign(moved.label-2)
        row={'kind':kind,'accuracy':float((z.pred==z.label).mean()),
             'opposite_direction_share_moved':float((d==-1).mean()),
             'flat_share':float((z.pred==2).mean())}
        for k in (0,4):
            sel=z[z.pred==k]
            row[f'pred{k}_n']=int(len(sel))
            row[f'pred{k}_precision']=float((sel.label==k).mean()) if len(sel) else None
        rows.append(row)
    diag=pd.DataFrame(rows).sort_values('kind')
    diag.to_csv(OUT/'cost_decode_diagnostics.csv',index=False)
    res['label_marginal']=(np.bincount(label,minlength=5)/len(label)).round(5).tolist()
    res['diagnostics']=diag.to_dict('records')
    res['reading']=('lgb_cost (tau=0) has the best accuracy and the fewest opposite-direction '
                    'calls but the worst score, because it answers flat on most rows and the '
                    'score pays nothing for a correct flat. lgb_ratio does the reverse: it beats '
                    'the same model under the earlier quantile decoding while getting fewer rows '
                    'right, by making many low-precision crash calls that inflate the chance '
                    'denominator. Neither is a better forecast than B0.')
    (OUT/'cost_decode_decision.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
    print(diag.round(4).to_string(index=False))

if __name__=='__main__':main()
