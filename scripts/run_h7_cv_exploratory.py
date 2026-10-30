#!/usr/bin/env python3
"""H7 exploratory: 5x5-fold CV on SDOH-NLI train (W->S, S->S, W+S->S, rules). Paths are this environment's; edit as needed."""
import sys, json; sys.path[:0]=['/home/user/finetune','/home/user/finetune/scripts']
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from run_h7_ood import load_sdoh, add_evidence, fit_predict
from run_wright_smoke import load, assign_splits
from src.status_rules import rule_status
from src.metrics import metrics_payload
w=load(Path('/home/user/ext/smokingbenchmark')); w['split']=assign_splits(w); w=add_evidence(w[w.split=='train'])
s=add_evidence(load_sdoh(Path('/home/user/ext/datasets/SDOH-NLI'),'train')).reset_index(drop=True)
y=s.label_native.values
scores={k:[] for k in ('W->S','S->S','W+S->S','Rules')}
pcls={k:[] for k in scores}
rk=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=0)
for tr,te in rk.split(s,y):
    a,b=s.iloc[tr],s.iloc[te]
    both=pd.concat([w[['evid0','label_native']],a[['evid0','label_native']]],ignore_index=True)
    preds={'W->S':fit_predict(w,b),'S->S':fit_predict(a,b),'W+S->S':fit_predict(both,b),'Rules':np.array([rule_status(e) for e in b.evid0],dtype=object)}
    for k,p in preds.items():
        m=metrics_payload(b.label_native.values,p); scores[k].append(m['macro_f1']); pcls[k].append({c:v['f1'] for c,v in m['per_class'].items()})
out={}
for k in scores:
    v=np.array(scores[k]); pc=pd.DataFrame(pcls[k]).mean().round(3).to_dict()
    out[k]={'macro_f1_mean':round(v.mean(),3),'sd':round(v.std(),3),'per_class_mean':pc}
    print(f"{k:8s} macroF1 {v.mean():.3f} ± {v.std():.3f}  per-class {pc}")
d=np.array(scores['W+S->S'])-np.array(scores['S->S']); print('W+S minus S: mean %.3f, folds W+S better: %d/25'%(d.mean(),(d>0).sum()))
out['W+S_minus_S']={'mean':round(d.mean(),3),'folds_better':int((d>0).sum()),'folds':25}
json.dump(out,open('/home/user/finetune/experiments/h7-ood-sdoh-nli/results/h7_exploratory_cv_sdoh_train.json','w'),indent=2)
