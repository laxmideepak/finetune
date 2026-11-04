import sys, json, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
tok,_=H._load(); W=H.weights(); U=W['lm_head.weight']; g=W['decoder.final_layer_norm.weight']
i={a:tok(a).input_ids[0] for a in ['current smoker','former smoker','never smoked','unknown']}
d=(U[i['never smoked']]-U[i['unknown']])*g; dc=(U[i['current smoker']]-U[i['unknown']])*g; df=(U[i['former smoker']]-U[i['unknown']])*g
wc=np.load(f'{sys.argv[1]}/w_centered.npy')
dd=H.dev_data(); m=dd.evid0.str.len().to_numpy()>0; ev=dd[m].reset_index(drop=True)
for mult in tuple(float(x) for x in sys.argv[2].split(',')):
    beta=31.26*mult
    H.note(f"Turn 11: selective edit at x{mult}: the value direction is made orthogonal to the current-unknown and former-unknown answer directions, so the change can only move never vs unknown.")
    for name,t in L.build_selective(W,wc,d,[dc,df],beta).items(): H.save_edit(name,t,f'turn 11 selective edit x{mult}')
    p=np.array(H.forward(ev.evid0.tolist(), which='edited', why=f'selective edit x{mult}')['edited'],dtype=object)
    full=np.full(len(dd),'UNKNOWN',dtype=object); full[m]=p
    print(mult, json.dumps(H.report(dd.label_native.to_numpy(),full)))
    json.dump({'beta':beta,'pred':list(p)},open(f'{sys.argv[1]}/t10_x{mult}.json','w'))
