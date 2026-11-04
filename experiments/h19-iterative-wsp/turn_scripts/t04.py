import sys, json, pickle, numpy as np, collections; sys.path.insert(0,'/home/user/finetune/scripts')
import wsp_harness as H
H.note("Turn 4 (analysis only, no pass): per-head decomposition of decoder block 19/22 cross-attention along d, and which input tokens the strongest heads attend to.")
A=pickle.load(open(f'{sys.argv[1]}/act03.pkl','rb')); a2=np.load(f'{sys.argv[1]}/act02.npz')
d=np.load(f'{sys.argv[1]}/d_never_unknown.npy'); pre=a2['decoder_final_layer_norm']; rms=np.sqrt((pre**2).mean(1))
W=H.weights(); tok,_=H._load(); G=np.array(A['groups'])
from src.flan_status import PROMPT
for blk,qk in ((19,'q19'),(22,'q22')):
    p=f'decoder.block.{blk}.layer.1.EncDecAttention.'
    Wq,Wk,Wv,Wo=W[p+'q.weight'],W[p+'k.weight'],W[p+'v.weight'],W[p+'o.weight']; nh=16; dk=64
    contrib=np.zeros((len(G),nh)); top=collections.defaultdict(collections.Counter)
    for n,(e,q) in enumerate(zip(A['enc'],A[qk])):
        Q=(Wq@q).reshape(nh,dk); K=(e@Wk.T).reshape(len(e),nh,dk); V=(e@Wv.T).reshape(len(e),nh,dk)
        s=np.einsum('hd,thd->ht',Q,K); s-=s.max(1,keepdims=True); att=np.exp(s); att/=att.sum(1,keepdims=True)
        o=np.einsum('ht,thd->hd',att,V)
        for h in range(nh): contrib[n,h]=(Wo[:,h*dk:(h+1)*dk]@o[h])@d/rms[n]
        toks=tok.convert_ids_to_tokens(tok(PROMPT.format(text=A['texts'][n]),truncation=True,max_length=256).input_ids)
        for h in range(nh): top[(G[n],h)][toks[int(att[h].argmax())]]+=1
    print(f'block {blk}: heads by (denial_wrong - denial_right) push along d')
    m={g:contrib[G==g].mean(0) for g in ('denial_wrong','denial_right','never_true')}
    order=np.argsort(-(m['denial_wrong']-m['denial_right']))
    for h in order[:6]:
        print(f'  head {h:2d}: wrong {m["denial_wrong"][h]:6.2f} right {m["denial_right"][h]:6.2f} never {m["never_true"][h]:6.2f} | top tokens wrong {top[("denial_wrong",h)].most_common(4)} never {top[("never_true",h)].most_common(4)}')
    np.save(f'{sys.argv[1]}/contrib{blk}.npy',contrib)
