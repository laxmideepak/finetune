import sys, json, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
wc=np.load(f'{sys.argv[1]}/w_centered.npy'); d=np.load(f'{sys.argv[1]}/d_never_unknown.npy'); W=H.weights()
dd=H.dev_data(); m=dd.evid0.str.len().to_numpy()>0; ev=dd[m].reset_index(drop=True)
for mult in tuple(float(x) for x in sys.argv[2].split(",")):
    beta=31.26*mult
    H.note(f"Turn 9: same centred edit at {mult}x strength (beta={beta:.1f}); does a stronger push move the near-zero denials without breaking true never?")
    for name,t in L.build(W,wc,d,beta).items(): H.save_edit(name,t,f'turn 9 centred edit x{mult}')
    p=np.array(H.forward(ev.evid0.tolist(), which='edited', why=f'centred edit x{mult}')['edited'],dtype=object)
    full=np.full(len(dd),'UNKNOWN',dtype=object); full[m]=p
    r=H.report(dd.label_native.to_numpy(),full); print(mult, json.dumps(r))
    json.dump({'beta':beta,'pred':list(p)},open(f'{sys.argv[1]}/t08_x{mult}.json','w'))
