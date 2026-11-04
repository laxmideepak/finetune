import sys, pickle, numpy as np; sys.path.insert(0,'/home/user/finetune/scripts')
import wsp_harness as H
from src.flan_status import PROMPT
from sklearn.metrics import roc_auc_score
H.note("Turn 5 (analysis only): at the encoder state of the 'never' OPTION token (last '▁never' in the prompt), is there a direction separating denial-only from true-never premises? Check mean-difference direction, its in-sample AUC and a leave-one-out version.")
A=pickle.load(open(f'{sys.argv[1]}/act03.pkl','rb')); tok,_=H._load(); G=np.array(A['groups'])
E=[]
for t,e in zip(A['texts'],A['enc']):
    toks=tok.convert_ids_to_tokens(tok(PROMPT.format(text=t),truncation=True,max_length=256).input_ids)
    i=max(j for j,x in enumerate(toks) if x=='▁never'); E.append(e[i])
E=np.array(E); den=np.isin(G,['denial_wrong','denial_right']); nev=G=='never_true'
w=E[den].mean(0)-E[nev].mean(0)
print('in-sample AUC (denial vs never) of w.e:', round(roc_auc_score(den[den|nev], (E@w)[den|nev]),3))
loo=[]
idx=np.where(den|nev)[0]
for i in idx:
    m=np.ones(len(E),bool); m[i]=False
    wi=E[den&m].mean(0)-E[nev&m].mean(0); loo.append(E[i]@wi - 0.5*((E[den&m]@wi).mean()+(E[nev&m]@wi).mean()))
loo=np.array(loo); print('leave-one-out accuracy (sign):', round(float(((loo>0)==den[idx]).mean()),3), 'denial correct', round(float((loo[den[idx]]>0).mean()),3), 'never correct', round(float((loo[~den[idx]]<=0).mean()),3))
print('projection means: denial_wrong', round(float((E[G=="denial_wrong"]@w).mean()),2), 'denial_right', round(float((E[G=="denial_right"]@w).mean()),2), 'never', round(float((E[nev]@w).mean()),2), '| sd', round(float((E[den|nev]@w).std()),2))
np.save(f'{sys.argv[1]}/w_option_never.npy', w); np.save(f'{sys.argv[1]}/E_never_opt.npy',E)
