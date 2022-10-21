import torch
import torch.nn as nn
import numpy as np

def eval_model(model, valid_dl):
    loss = model.loss.unit_loss
    model.eval()

    LLsum, Tsum, Rsum = 0, 0, 0
    from tqdm import tqdm
        
    device = next(model.parameters()).device  # device the model is on
    if isinstance(valid_dl, dict):
        for dsub in valid_dl.keys():
                if valid_dl[dsub].device != device:
                    valid_dl[dsub] = valid_dl[dsub].to(device)
        rpred = model(valid_dl)
        LLsum = loss(rpred,
                    valid_dl['robs'][:,model.cids],
                    data_filters=valid_dl['dfs'][:,model.cids],
                    temporal_normalize=False)
        Tsum = valid_dl['dfs'][:,model.cids].sum(dim=0)
        Rsum = (valid_dl['dfs'][:,model.cids]*valid_dl['robs'][:,model.cids]).sum(dim=0)

    else:
        for data in tqdm(valid_dl, desc='Eval models'):
                    
            for dsub in data.keys():
                if data[dsub].device != device:
                    data[dsub] = data[dsub].to(device)
            
            with torch.no_grad():
                rpred = model(data)
                LLsum += loss(rpred,
                        data['robs'][:,model.cids],
                        data_filters=data['dfs'][:,model.cids],
                        temporal_normalize=False)
                Tsum += data['dfs'][:,model.cids].sum(dim=0)
                Rsum += (data['dfs'][:,model.cids] * data['robs'][:,model.cids]).sum(dim=0)
                
    LLneuron = LLsum/Rsum.clamp(1)

    rbar = Rsum/Tsum.clamp(1)
    LLnulls = torch.log(rbar)-1
    LLneuron = -LLneuron - LLnulls

    LLneuron/=np.log(2)

    return LLneuron.detach().cpu().numpy()