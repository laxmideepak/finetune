import sys, json, pickle, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
from src.flan_status import PROMPT
tok,_=H._load(); W=H.weights(); U=W['lm_head.weight']; g=W['decoder.final_layer_norm.weight']
i={a:tok(a).input_ids[0] for a in ['current smoker','former smoker','never smoked','unknown']}
d=(U[i['never smoked']]-U[i['unknown']])*g; dc=(U[i['current smoker']]-U[i['unknown']])*g; df=(U[i['former smoker']]-U[i['unknown']])*g
E=np.load(f'{sys.argv[1]}/E_never_opt.npy'); G=np.array(pickle.load(open(f'{sys.argv[1]}/act03.pkl','rb'))['groups'])
den=np.isin(G,['denial_wrong','denial_right']); nev=G=='never_true'
X=np.vstack([E[den]-E[den].mean(0),E[nev]-E[nev].mean(0)]); Sg=X.T@X/len(X); lam=0.5
Sg=(1-lam)*Sg+lam*np.trace(Sg)/Sg.shape[0]*np.eye(Sg.shape[0]); w=np.linalg.solve(Sg,E[den].mean(0)-E[nev].mean(0))
wc=np.load(f'{sys.argv[1]}/w_centered.npy')  # reuse the stored N/A + mean basis by recomputing: orthogonalise w against mean(E) and the N/A state
# N/A state was folded into w_centered; recover the basis from turn 7 inputs is not stored, so orthogonalise against mean(E) and the turn-7 component wc-w0 is unavailable: re-derive N/A from the log is not possible without a pass -> use the projection of w_centered's removed part
w0=np.load(f'{sys.argv[1]}/w_option_never.npy'); removed=w0-wc
B=np.stack([E.mean(0), removed]); Q,_=np.linalg.qr(B.T); wl=w-Q@(Q.T@w)
pr=E@(wl/np.linalg.norm(wl)); print('LDA centred proj: denial_wrong',round(pr[G=="denial_wrong"].mean(),3),'denial_right',round(pr[G=="denial_right"].mean(),3),'never',round(pr[nev].mean(),3),'sd',round(pr.std(),3))
np.save(f'{sys.argv[1]}/w_lda_centered.npy',wl)
gain=np.mean([np.linalg.norm(W[f'decoder.block.{b}.layer.1.EncDecAttention.o.weight'][:,h*64:(h+1)*64].T@d) for b,h in L.HEADS])
base_beta=1.5*833.89/(3*gain*pr.std())
dd=H.dev_data(); m=dd.evid0.str.len().to_numpy()>0; ev=dd[m].reset_index(drop=True)
for mult in tuple(float(x) for x in sys.argv[2].split(',')):
    beta=base_beta*mult
    H.note(f"Turn 13: selective edit with the shrinkage-discriminant direction (lambda 0.5, centred) at x{mult} (beta={beta:.1f}).")
    for name,t in L.build_selective(W,wl,d,[dc,df],beta).items(): H.save_edit(name,t,f'turn 13 LDA selective edit x{mult}')
    p=np.array(H.forward(ev.evid0.tolist(), which='edited', why=f'LDA selective edit x{mult}')['edited'],dtype=object)
    full=np.full(len(dd),'UNKNOWN',dtype=object); full[m]=p
    print(mult, round(beta,1), json.dumps(H.report(dd.label_native.to_numpy(),full)))
    json.dump({'beta':beta,'pred':list(p)},open(f'{sys.argv[1]}/t13_x{mult}.json','w'))
