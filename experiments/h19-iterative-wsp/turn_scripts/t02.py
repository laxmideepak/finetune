import sys, json, numpy as np, torch; sys.path.insert(0,'/home/user/finetune/scripts')
import wsp_harness as H
H.note("Turn 2: find which decoder parts push 'never smoked' over 'unknown'. Read the answer tokens and the output-matrix direction d = W_U[never] - W_U[unknown] from the weights; then one activation pass records, for denial-only, true-never and correctly-unknown denials, each decoder sublayer's output at the first answer position, so I can see which sublayers write along d.")
tok,_=H._load()
for a in ['current smoker','former smoker','never smoked','unknown']: print(a, tok(a).input_ids, tok.convert_ids_to_tokens(tok(a).input_ids))
W=H.weights(); U=W['lm_head.weight']; g=W['decoder.final_layer_norm.weight']
ids={a:tok(a).input_ids[0] for a in ['current smoker','former smoker','never smoked','unknown']}
d=(U[ids['never smoked']]-U[ids['unknown']])*g   # direction in pre-norm residual (norm scale folded in)
np.save('/tmp/claude-0/-home-user-finetune/0bfd38ec-7533-5c32-8deb-0b20c280185a/scratchpad/h19/d_never_unknown.npy', d)
base=json.load(open('/tmp/claude-0/-home-user-finetune/0bfd38ec-7533-5c32-8deb-0b20c280185a/scratchpad/h19/base_preds.json'))
grp={'denial_wrong':[r['evid0'] for r in base if r['reason']=='denial-only' and r['pred']=='NEVER'],
     'denial_right':[r['evid0'] for r in base if r['reason']=='denial-only' and r['pred']=='UNKNOWN'],
     'never_true':[r['evid0'] for r in base if r['label_native']=='NEVER']}
texts=sum(grp.values(),[]); print({k:len(v) for k,v in grp.items()})
hooks={}
for i in range(24):
    for nm,mod in (('sa',f'decoder.block.{i}.layer.0.SelfAttention'),('ca',f'decoder.block.{i}.layer.1.EncDecAttention'),('ff',f'decoder.block.{i}.layer.2.DenseReluDense')):
        hooks[mod]=lambda inp,out: (out[0] if isinstance(out,tuple) else out)[0,0].double().numpy()
hooks['decoder.final_layer_norm']=lambda inp,out: inp[0][0,0].double().numpy()
out=H.forward(texts, which='original', mode='activations', hooks=hooks, why='decoder sublayer outputs at first answer position for denial (wrong/right) vs true never')['original']
np.savez_compressed('/tmp/claude-0/-home-user-finetune/0bfd38ec-7533-5c32-8deb-0b20c280185a/scratchpad/h19/act02.npz', **{k.replace('.','_'):np.stack(v) for k,v in out.items()}, groups=np.array(sum([[k]*len(v) for k,v in grp.items()],[])))
G=np.array(sum([[k]*len(v) for k,v in grp.items()],[]))
pre=np.stack(out['decoder.final_layer_norm']); rms=np.sqrt((pre**2).mean(1))
print('logit-diff proxy (never-unknown) by group:', {k: round(float(((pre[G==k]@d)/rms[G==k]).mean()),2) for k in grp})
rows=[]
for mod in hooks:
    if mod=='decoder.final_layer_norm': continue
    c=np.stack(out[mod])@d/rms
    rows.append((mod, c[G=='denial_wrong'].mean(), c[G=='denial_right'].mean(), c[G=='never_true'].mean()))
rows.sort(key=lambda r: -(r[1]-r[2]))
print('top sublayers: contribution along d (denial_wrong, denial_right, never_true)')
for r in rows[:12]: print(f'{r[0]:45s} {r[1]:7.2f} {r[2]:7.2f} {r[3]:7.2f}')
