import sys, json, numpy as np; sys.path.insert(0,'/home/user/finetune/scripts')
import wsp_harness as H
H.note("Turn 1: baseline. Run the unedited model on all development premises that have evidence, to see which classes it confuses and save per-item predictions for the analysis.")
d=H.dev_data(); ev=d[d.evid0.str.len()>0].reset_index(drop=True)
print('dev premises',len(d),'with evidence',len(ev), ev.label_native.value_counts().to_dict(), ev.reason.value_counts(dropna=False).to_dict())
out=H.forward(ev.evid0.tolist(), which='original', why='baseline predictions on development evidence')
p=np.array(out['original'],dtype=object)
full=np.full(len(d),'UNKNOWN',dtype=object); full[d.evid0.str.len().to_numpy()>0]=p
print(json.dumps(H.report(d.label_native.to_numpy(), full)))
ev['pred']=p; ev[['evid0','label_native','reason','pred']].to_json('/tmp/claude-0/-home-user-finetune/0bfd38ec-7533-5c32-8deb-0b20c280185a/scratchpad/h19/base_preds.json',orient='records',indent=0)
import collections; print(collections.Counter(zip(ev.label_native,ev.reason.fillna('-'),ev.pred)).most_common(20))
