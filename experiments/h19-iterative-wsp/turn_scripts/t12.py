import sys, pickle, numpy as np; sys.path[:0]=['/home/user/finetune/scripts', sys.argv[1]]
import wsp_harness as H
from sklearn.metrics import roc_auc_score
H.note("Turn 12 (analysis only): compare the mean-difference direction with closed-form shrinkage discriminant directions at the 'never' option token, by leave-one-out separation of denial vs never (no model run).")
E=np.load(f'{sys.argv[1]}/E_never_opt.npy'); G=np.array(pickle.load(open(f'{sys.argv[1]}/act03.pkl','rb'))['groups'])
den=np.isin(G,['denial_wrong','denial_right']); nev=G=='never_true'; idx=np.where(den|nev)[0]
def direction(Xd,Xn,lam):
    mu=Xd.mean(0)-Xn.mean(0)
    if lam is None: return mu
    X=np.vstack([Xd-Xd.mean(0),Xn-Xn.mean(0)]); S=X.T@X/len(X); S=(1-lam)*S+lam*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    return np.linalg.solve(S,mu)
for lam in (None,0.9,0.5,0.2,0.05):
    sc=[];
    for i in idx:
        m=np.ones(len(E),bool); m[i]=False
        w=direction(E[den&m],E[nev&m],lam); t=0.5*((E[den&m]@w).mean()+(E[nev&m]@w).mean())
        sc.append(E[i]@w-t)
    sc=np.array(sc); y=den[idx]
    print(f'lambda={lam}: LOO AUC {roc_auc_score(y,sc):.3f}  acc {((sc>0)==y).mean():.3f}  denial_wrong separated {(sc[G[idx]=="denial_wrong"]>0).mean():.3f}  never kept {(sc[~y]<=0).mean():.3f}')
