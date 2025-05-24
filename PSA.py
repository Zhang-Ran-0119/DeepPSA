from DataProcessing.data_process import Data_Process
from modules.model_GCN import Protac_GCN
from modules.model_WLN import Protac_WLN
from modules.model_GIN import Protac_GIN
from modules.model_GATv2 import Protac_GATv2
from modules.model_AttentionFP import Protac_AFP
from modules.model_GAT import Protac_GAT
from modules.model_GateGCN import Protac_GateGCN
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
import dgl
from judge_function import metric_functions as f
import sys


def test(model, validloader, device):
    model.to(device)
    score = []
    score2 = []
    labels = []
    smiles = []
    label_pre = []
    label_true = []
    n=0
    nn = 1
    with torch.no_grad():
        model.eval()
        for molecule, label, smile in validloader:
            output = model(molecule.to(device))
            output = torch.nn.functional.softmax(output)
            score = score + output[:, 1].cpu().tolist()
            score2 = score2 + output[:, 0].cpu().tolist()
            # score = score + torch.nn.functional.softmax(output, 1)[:, 1].cpu().tolist()
            labels = labels + label.cpu().tolist()
            label_pre = label_pre + torch.max(output, 1)[1].cpu().tolist()
            smiles.extend(smile)
    df = pd.DataFrame({
        'smiles': smiles,
        'es': score2,
        'hs': score,
        'lable_pre': label_pre
    })
    df.to_csv('result.csv', mode='w', header=True, index=False)



def _collate_fn(batch):
    graphs, labels ,smiles= zip(*batch)
    g = dgl.batch(graphs)
    global bg
    bg=g
    labels = torch.tensor(labels, dtype=torch.long)
    return g, labels,smiles

BATCH_SIZE = 64
test_data=sys.argv[1]


test_dataset = Data_Process(path=test_data, name='test', save_dir='data_process/mytest', verbose=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_fn)


PSA_models = {
    'GCN': Protac_GCN(74,[256,512,1024,2048],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropouts=[0.2,0.5,0.5,0.5]),
    'WLN':Protac_WLN(74,13,3),
    'GIN':Protac_GIN(74,3),
    'GAT':Protac_GAT(74,[128,512,2048],[2,2,2],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropout=[0,0,0]),
    'GATv2':Protac_GATv2(74,[128,512,2048],[2,2,2],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropout=[0,0,0]),
    'AttentiveFP':Protac_AFP(74,13,3),
    'GatedGCN':Protac_GateGCN(74,13,2048),
}
network=PSA_models['GCN']

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

state_dict = torch.load('PSA/FinetuneModel.pth')
current_state_dict = network.state_dict()
for name, param in state_dict.items():
    if name in current_state_dict:
        current_state_dict[name].copy_(param)
    else:
        print(f"Skipping {name} as it does not exist in the current model.")
network.load_state_dict(current_state_dict, strict=False)
network.to(device)
test(network,test_loader,device=device)










