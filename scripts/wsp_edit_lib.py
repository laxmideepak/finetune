import numpy as np
HEADS=((19,1),(19,7),(22,1))
def build(W, w, d, beta, heads=HEADS, dk=64):
    """Rank-1 change per head: dWv_h = -beta * a_h w_hat^T, a_h = unit(Wo_h^T d). Returns {name: new tensor}."""
    w_hat=w/np.linalg.norm(w); new={}
    for blk,h in heads:
        p=f'decoder.block.{blk}.layer.1.EncDecAttention.'
        Wv=new.get(p+'v.weight', W[p+'v.weight'].copy()); Wo=W[p+'o.weight']
        a=Wo[:,h*dk:(h+1)*dk].T@d; a/=np.linalg.norm(a)
        Wv[h*dk:(h+1)*dk,:] -= beta*np.outer(a,w_hat)
        new[p+'v.weight']=Wv
    return new

def build_selective(W, w, d, d_others, beta, heads=HEADS, dk=64):
    """Like build(), but a_h is made orthogonal (in output space) to the other answer directions, so
    the change moves only never-vs-unknown: Wo_h a_h . d_other = 0 for each d_other."""
    w_hat=w/np.linalg.norm(w); new={}
    for blk,h in heads:
        p=f'decoder.block.{blk}.layer.1.EncDecAttention.'
        Wv=new.get(p+'v.weight', W[p+'v.weight'].copy()); Wo=W[p+'o.weight'][:,h*dk:(h+1)*dk]
        a=Wo.T@d
        C=np.stack([Wo.T@x for x in d_others],1); Q,_=np.linalg.qr(C); a=a-Q@(Q.T@a); a/=np.linalg.norm(a)
        Wv[h*dk:(h+1)*dk,:] -= beta*np.outer(a,w_hat)
        new[p+'v.weight']=Wv
    return new
