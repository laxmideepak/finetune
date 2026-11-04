import sys, json, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
from src.flan_status import PROMPT
tok,_=H._load(); W=H.weights(); U=W['lm_head.weight']; g=W['decoder.final_layer_norm.weight']
i={a:tok(a).input_ids[0] for a in ['current smoker','former smoker','never smoked','unknown']}
d=(U[i['never smoked']]-U[i['unknown']])*g; dc=(U[i['current smoker']]-U[i['unknown']])*g; df=(U[i['former smoker']]-U[i['unknown']])*g
beta=json.load(open(f'{sys.argv[1]}/t15_x4.0.json'))['beta']; wn=np.load(f'{sys.argv[1]}/w_lda_neutral.npy')
for name,t in L.build_selective(W,wn,d,[dc,df],beta).items(): H.save_edit(name,t,'turn 16: keep the neutral discriminant edit at x4 (best on development)')
H.note("Turn 16: keep edit 1 (x4). Second failure: CURRENT premises read as FORMER. One activation pass on CURRENT and FORMER development premises with edit 1 applied: decoder sublayer outputs at the first answer position, the cross-attention query inputs, and per-token encoder states, to find which heads push 'former' over 'current' and what they read.")
d2=(U[i['former smoker']]-U[i['current smoker']])*g; np.save(f'{sys.argv[1]}/d_former_current.npy',d2)
base=json.load(open(f'{sys.argv[1]}/base_preds.json')); p15=json.load(open(f'{sys.argv[1]}/t15_x4.0.json'))['pred']
ev=[dict(r,pred1=p) for r,p in zip(base,p15)]
grp={'cur_wrong':[r['evid0'] for r in ev if r['label_native']=='CURRENT' and r['pred1']=='FORMER'],
     'cur_right':[r['evid0'] for r in ev if r['label_native']=='CURRENT' and r['pred1']=='CURRENT'],
     'former_true':[r['evid0'] for r in ev if r['label_native']=='FORMER']}
print({k:len(v) for k,v in grp.items()}); texts=sum(grp.values(),[])
for k in ('cur_wrong',): print('examples:', [t[:90] for t in grp[k][:8]])
hooks={'decoder.final_layer_norm': lambda i_,o_: i_[0][0,0].double().numpy(), 'encoder.final_layer_norm': lambda i_,o_: o_[0].double().numpy()}
for b in range(16,24):
    hooks[f'decoder.block.{b}.layer.1.EncDecAttention']=lambda i_,o_: o_[0][0,0].double().numpy()
    hooks[f'decoder.block.{b}.layer.1.layer_norm']=lambda i_,o_: o_[0,0].double().numpy()
    hooks[f'decoder.block.{b}.layer.2.DenseReluDense']=lambda i_,o_: o_[0,0].double().numpy()
out=H.forward(texts, which='edited', mode='activations', hooks=hooks, why='current-vs-former analysis with edit 1 applied')['edited']
import pickle; pickle.dump({'texts':texts,'groups':sum([[k]*len(v) for k,v in grp.items()],[]),**out},open(f'{sys.argv[1]}/act16.pkl','wb'))
G=np.array(sum([[k]*len(v) for k,v in grp.items()],[])); pre=np.stack(out['decoder.final_layer_norm']); rms=np.sqrt((pre**2).mean(1))
rows=[]
for k,v in out.items():
    if 'EncDecAttention' in k or 'DenseReluDense' in k:
        c=np.stack(v)@d2/rms; rows.append((k,c[G=='cur_wrong'].mean(),c[G=='cur_right'].mean(),c[G=='former_true'].mean()))
rows.sort(key=lambda r: -(r[1]-r[2]))
for r in rows[:8]: print(f'{r[0]:45s} wrong {r[1]:6.2f} right {r[2]:6.2f} former {r[3]:6.2f}')
