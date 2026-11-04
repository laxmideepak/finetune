import sys, json, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
from src.flan_status import PROMPT
H.note("Turn 7: the turn-6 edit moved the calibration reference ('N/A' input) too, so the effect cancelled. Record the encoder state at the 'never' option token for the 'N/A' input (1 text), then orthogonalise the direction against the mean option state and the N/A state so the edit only measures differences between sentences. Re-test on development premises.")
tok,_=H._load()
o=H.forward(['N/A'], which='original', mode='activations', hooks={'encoder.final_layer_norm':lambda i,o_: o_[0].double().numpy()}, why='encoder state for calibration input N/A')['original']['encoder.final_layer_norm'][0]
toks=tok.convert_ids_to_tokens(tok(PROMPT.format(text='N/A')).input_ids); na=o[max(j for j,x in enumerate(toks) if x=='▁never')]
E=np.load(f'{sys.argv[1]}/E_never_opt.npy'); w=np.load(f'{sys.argv[1]}/w_option_never.npy'); d=np.load(f'{sys.argv[1]}/d_never_unknown.npy')
B=np.stack([E.mean(0), na]); Q,_=np.linalg.qr(B.T)
wc=w-Q@(Q.T@w); wh=wc/np.linalg.norm(wc); np.save(f'{sys.argv[1]}/w_centered.npy',wc)
pr=E@wh; G=np.array(__import__('pickle').load(open(f'{sys.argv[1]}/act03.pkl','rb'))['groups'])
print('centred projection means: denial_wrong',round(pr[G=="denial_wrong"].mean(),3),'denial_right',round(pr[G=="denial_right"].mean(),3),'never',round(pr[G=="never_true"].mean(),3),'| NA',round(float(na@wh),4),'sd',round(pr.std(),3))
W=H.weights(); gain=np.mean([np.linalg.norm(W[f'decoder.block.{b}.layer.1.EncDecAttention.o.weight'][:,h*64:(h+1)*64].T@d) for b,h in L.HEADS])
beta=1.5*833.89/(3*gain*pr.std()); print('beta',round(beta,2))
for name,t in L.build(W,wc,d,beta).items(): H.save_edit(name,t,'turn 7 centred rank-1 value edit')
dd=H.dev_data(); ev=dd[dd.evid0.str.len()>0].reset_index(drop=True)
p=np.array(H.forward(ev.evid0.tolist(), which='edited', why=f'centred edit beta={beta:.2f}')['edited'],dtype=object)
full=np.full(len(dd),'UNKNOWN',dtype=object); full[dd.evid0.str.len().to_numpy()>0]=p
print(json.dumps(H.report(dd.label_native.to_numpy(),full)))
json.dump({'beta':beta,'pred':list(p)},open(f'{sys.argv[1]}/t07_pred.json','w'))
