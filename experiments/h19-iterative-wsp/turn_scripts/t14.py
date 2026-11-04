import sys, json, pickle, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
from src.flan_status import PROMPT
H.note("Turn 14: record encoder states at the 'never' option token for CURRENT and FORMER development premises (1 pass), then make the discriminant direction neutral for them: remove its components along (mean_current - mean_all) and (mean_former - mean_all) as well as the shared and N/A components.")
tok,_=H._load(); dd=H.dev_data(); ev=dd[dd.evid0.str.len()>0]
cf=ev[ev.label_native.isin(['CURRENT','FORMER'])].reset_index(drop=True)
o=H.forward(cf.evid0.tolist(), which='original', mode='activations', hooks={'encoder.final_layer_norm':lambda i,o_: o_[0].double().numpy()}, why='never-option encoder states for CURRENT/FORMER premises')['original']['encoder.final_layer_norm']
Ecf=[]
for t,e in zip(cf.evid0,o):
    toks=tok.convert_ids_to_tokens(tok(PROMPT.format(text=t),truncation=True,max_length=256).input_ids); Ecf.append(e[max(j for j,x in enumerate(toks) if x=='▁never')])
Ecf=np.array(Ecf); np.save(f'{sys.argv[1]}/E_cf.npy',Ecf); np.save(f'{sys.argv[1]}/cf_labels.npy',cf.label_native.to_numpy())
E=np.load(f'{sys.argv[1]}/E_never_opt.npy'); wl=np.load(f'{sys.argv[1]}/w_lda_centered.npy')
allm=np.vstack([E,Ecf]).mean(0); lab=cf.label_native.to_numpy()
for name,w in (('before',wl),):
    wh=w/np.linalg.norm(w); print(name,'proj current',round(float((Ecf[lab=="CURRENT"]@wh).mean()),3),'former',round(float((Ecf[lab=="FORMER"]@wh).mean()),3))
w0=np.load(f'{sys.argv[1]}/w_option_never.npy'); removed=w0-np.load(f'{sys.argv[1]}/w_centered.npy')
B=np.stack([E.mean(0), removed, Ecf[lab=="CURRENT"].mean(0)-allm, Ecf[lab=="FORMER"].mean(0)-allm]); Q,_=np.linalg.qr(B.T)
wn=wl-Q@(Q.T@wl); wh=wn/np.linalg.norm(wn); np.save(f'{sys.argv[1]}/w_lda_neutral.npy',wn)
G=np.array(pickle.load(open(f'{sys.argv[1]}/act03.pkl','rb'))['groups'])
print('after: current',round(float((Ecf[lab=="CURRENT"]@wh).mean()),3),'former',round(float((Ecf[lab=="FORMER"]@wh).mean()),3),'denial_wrong',round(float((E[G=="denial_wrong"]@wh).mean()),3),'never',round(float((E[G=="never_true"]@wh).mean()),3),'sd',round(float((E@wh).std()),3))
