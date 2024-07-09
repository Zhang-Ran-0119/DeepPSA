from torch.utils.data import Dataset
import dgl
import torch

class Protac(Dataset):
    def __init__(self,molecule,label):
        super().__init__()
        self.molecule=molecule
        self.labels=label

    def __len__(self):
        return len(self.molecule)

    def __getitem__(self, index):
        sample = {
            "molecules":self.molecule[index],
            "labels":self.labels[index],
        }
        return sample


import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, smiles_list, label_list):
        self.smiles_list = smiles_list
        self.label_list = label_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        labels = self.label_list[index]
        return smiles, labels


def collate(data):
    molecule=[x['molecules'] for x in data]
    labels=[x['labels'] for x in data]
    batch={}
    batch["molecules"]=dgl.batch(molecule)
    batch['labels']=torch.tensor(labels)
    return batch

