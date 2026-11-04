import sys, json, numpy as np; sys.path.insert(0,'/home/user/finetune/scripts')
import wsp_harness as H
H.note("Turn 3: record per-token encoder outputs and the query inputs of decoder cross-attention in blocks 19 and 22 (first answer position) for the same 106 texts, so I can compute each head's attention and its push along d from the weights.")
tok,_=H._load(); base=json.load(open(f'{sys.argv[1]}/base_preds.json'))
grp={'denial_wrong':[r['evid0'] for r in base if r['reason']=='denial-only' and r['pred']=='NEVER'],
     'denial_right':[r['evid0'] for r in base if r['reason']=='denial-only' and r['pred']=='UNKNOWN'],
     'never_true':[r['evid0'] for r in base if r['label_native']=='NEVER']}
texts=sum(grp.values(),[])
hooks={'encoder.final_layer_norm': lambda i,o: o[0].double().numpy(),
       'decoder.block.19.layer.1.layer_norm': lambda i,o: o[0,0].double().numpy(),
       'decoder.block.22.layer.1.layer_norm': lambda i,o: o[0,0].double().numpy()}
out=H.forward(texts, which='original', mode='activations', hooks=hooks, why='per-token encoder states + cross-attn queries (blocks 19, 22)')['original']
import pickle; pickle.dump({'texts':texts,'groups':sum([[k]*len(v) for k,v in grp.items()],[]),'enc':out['encoder.final_layer_norm'],'q19':out['decoder.block.19.layer.1.layer_norm'],'q22':out['decoder.block.22.layer.1.layer_norm']},open(f'{sys.argv[1]}/act03.pkl','wb'))
print('saved', len(texts))
