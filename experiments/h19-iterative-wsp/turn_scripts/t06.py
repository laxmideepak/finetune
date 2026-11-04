import sys, json, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H, edit_lib as L
W=H.weights(); w=np.load(f'{sys.argv[1]}/w_option_never.npy'); d=np.load(f'{sys.argv[1]}/d_never_unknown.npy'); E=np.load(f'{sys.argv[1]}/E_never_opt.npy')
wh=w/np.linalg.norm(w); proj=E@wh; print('w_hat.e: mean', round(proj.mean(),3), 'sd', round(proj.std(),3))
gain=np.mean([np.linalg.norm(W[f'decoder.block.{b}.layer.1.EncDecAttention.o.weight'][:,h*64:(h+1)*64].T@d) for b,h in L.HEADS])
rms=np.sqrt((np.load(f'{sys.argv[1]}/act02.npz')['decoder_final_layer_norm']**2).mean(1)).mean()
beta=1.5*rms/(3*gain*proj.std())   # target: 1 sd of projection moves the never-vs-unknown logit by ~1.5 across the 3 heads
print('gain', round(float(gain),2),'rms',round(float(rms),2),'beta',round(float(beta),2))
H.note(f"Turn 6: rank-1 edit of the value weights of cross-attention heads 19.1, 19.7, 22.1: dWv_h = -beta a_h w^T with a_h the head-value direction that writes 'never minus unknown' and w the denial-vs-never direction read at the 'never' option token. beta={beta:.2f} set analytically so one s.d. of the projection shifts the never-unknown logit by about 1.5. Test original vs edited on all development premises with evidence.")
for name,t in L.build(W,w,d,beta).items(): H.save_edit(name,t,'turn 6 rank-1 value edit (heads 19.1, 19.7, 22.1)')
dd=H.dev_data(); ev=dd[dd.evid0.str.len()>0].reset_index(drop=True)
out=H.forward(ev.evid0.tolist(), which='edited', why=f'edited (beta={beta:.2f}) on all development evidence')
p=np.array(out['edited'],dtype=object); full=np.full(len(dd),'UNKNOWN',dtype=object); full[dd.evid0.str.len().to_numpy()>0]=p
print(json.dumps(H.report(dd.label_native.to_numpy(),full)))
json.dump({'beta':beta,'pred':list(p)},open(f'{sys.argv[1]}/t06_pred.json','w'))
