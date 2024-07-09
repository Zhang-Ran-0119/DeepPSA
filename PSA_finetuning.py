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
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from judge_function import metric_functions as f
import torch.nn.functional as F
import sys
import pandas as pd

import dgl
from torch.optim.lr_scheduler import CosineAnnealingLR

def valid(model, validloader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        label_true = []
        label_pre = []
        score = []
        loss = []
        iteration = 0
        for molecule,label,smile in validloader:
            label = label.to(device)
            output = model(molecule.to(device))
            loss_valid = criterion(output, label)
            loss.append(loss_valid.item())
            score = score + torch.nn.functional.softmax(output, 1)[:, 1].cpu().tolist()
            label_pre = label_pre + torch.max(output, 1)[1].cpu().tolist()
            label_true = label_true + label.cpu().tolist()
            iteration += 1
        model.train()
    return sum(loss)/iteration, f['ACC'](label_true,score), f['AUROC'](label_true,score)


def test(model, validloader, device):
    print("testing")
    label_true = []
    label_pre = []
    score = []
    score2=[]
    labels=[]
    loss = []
    smiles=[]
    iteration = 0
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        for molecule, label ,smile in validloader:
            output = []
            label = label.to(device)
            output = model(molecule.to(device))
            loss_valid = criterion(output, label)
            loss.append(loss_valid.item())
            output = torch.nn.functional.softmax(output)
            # score = score + output[:, 1].cpu().tolist()
            # score2 = score2 + output[:, 0].cpu().tolist()
            score = score + torch.nn.functional.softmax(output, 1)[:, 1].cpu().tolist()
            smiles.extend(smile)
            labels=labels+label.cpu().tolist()
            label_pre = label_pre + torch.max(output, 1)[1].cpu().tolist()
            label_true = label_true + label.cpu().tolist()
            iteration += 1
    df = pd.DataFrame({
        'smiles':smiles,
        #'es': score2,
        'hs': score,
        'label':labels
     })
    df.to_csv(save_path + '/test_out.csv', index=False)
    acc = f['ACC'](label_true,score)
    f1 = f['f1'](label_true,score)
    roc_auc = f['AUROC'](label_true,score)
    precision = f['precision'](label_true,score)
    recall=f['recall'](label_true, score)
    specificity=f['specificity'](label_true, score)
    sensitivity = f['sensitivity'](label_true, score)
    threshold=0.5
    print("acc:",acc)
    print("f1:", f1)
    print("roc_auc:", roc_auc)
    print("precision:",precision)
    print("recall:",recall)
    print("specificity:",specificity)
    print("sensitivity:",sensitivity)
    print("threshold:",threshold)



def train(model,batch_size, device, trainloader, validloader, lr, epoch,loss_name,writer,pretrain_model_path,optimizer_path):
    model = model.to(device)
    model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_state = torch.load(optimizer_path, map_location=device)
    opt.load_state_dict(optimizer_state)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    batch_id = 0
    cos_lr = CosineAnnealingLR(optimizer=opt, T_max=150, eta_min=0)
    _,_,_ = valid(model, validloader, device)
    criterion = nn.CrossEntropyLoss()
    best_val_roc=0
    now_roc=0
    temp=30
    global_id=0
    flag=0
    counter = 0
    for i in range(epoch):
        total_num=0
        print("training...", i)
        for molecule,label,smile in tqdm(train_loader, desc=f'Epoch {i + 1}/{epoch}', unit='batch'):
            batch_id+=1
            current_lr = opt.param_groups[0]['lr']
            print("Current learning rate:", current_lr)
            label = label.to(device)
            output = model(molecule.to(device))
            loss = criterion(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_num += batch_size
            cos_lr.step()
            #每隔 batch_to_log 批次记录一次损失
            if batch_id % batch_to_log == 0:
                global_id += 1
                val_loss, val_acc, auroc = valid(model, validloader, device)
                print('Train epoch %d, train_loss: %.4f, test_loss:%.4f, test_auroc:%.4f' % (
                    i, loss.item(), val_loss, auroc))
                writer.add_scalar('test_loss', val_loss, global_step=global_id)
                writer.add_scalar('test_auroc', auroc, global_step=global_id)
                if auroc > best_val_roc:
                    best_val_roc = auroc
                    counter = 0
                else:
                    counter += 1
                if counter >= 5:
                    print("Early stopping. Training stopped.")
                    test(model, test_loader, device,)
                    return model, opt
    print("Training stopped.")
    test(model, test_loader, device, 0)
    return model,opt

def _collate_fn(batch):
    graphs, labels ,smiles = zip(*batch)
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.long)
    return g, labels,smiles


BATCH_SIZE = 256
EPOCH = 200
LEARNING_RATE=1e-5
WARMUP_STEPS=30





train_data=sys.argv[2]
valid_data = sys.argv[3]
test_data=sys.argv[4]
model_name=sys.argv[1]


print("preparing for train data...")
train_dataset = Data_Process(path=train_data, name='train', save_dir='data_process/test', verbose=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_fn)
print("preparing for valid data...")
valid_dataset = Data_Process(path=valid_data, name='valid', save_dir='data_process/test', verbose=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_fn)
print("preparing for test data...")
test_dataset= Data_Process(path=test_data, name='test', save_dir='data_process/test', verbose=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_fn)
batch_to_log = 50


writer = SummaryWriter('log_directory')

PSA_models = {
    'GCN': Protac_GCN(74,[256,512,1024,2048],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropouts=[0.2,0.5,0.5,0.5]),
    'WLN':Protac_WLN(74,13,3),
    'GIN':Protac_GIN(74,3),
    'GAT':Protac_GAT(74,[128,512,2048],[2,2,2],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropout=[0,0,0]),
    'GATv2':Protac_GATv2(74,[128,512,2048],[2,2,2],activation=[F.leaky_relu_,F.leaky_relu_,F.leaky_relu_],dropout=[0,0,0]),
    'AttentiveFP':Protac_AFP(74,13,3),
    'GatedGCN':Protac_GateGCN(74,13,2048),
}

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

network=PSA_models['GCN']
save_path=model_name
pretrained_model_path="PSA/Model.pth"
optimizer_path="PSA/Optimizer.pth"
network.to(device)
model,optimizer=train(
    model=network,
    batch_size=BATCH_SIZE,
    device=device,
    trainloader=train_loader,
    validloader=valid_loader,
    lr=LEARNING_RATE,
    epoch=EPOCH,
    loss_name="test",
    writer=writer,
    pretrain_model_path=pretrained_model_path,
    optimizer_path=optimizer_path
    )
torch.save(model.state_dict(), save_path+'/FinetuneModel.pth')
torch.save(optimizer.state_dict(), save_path+'/FinetuneOptimizer.pth')



