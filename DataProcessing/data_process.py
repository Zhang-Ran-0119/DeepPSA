import pandas as pd
import torch
from functools import partial
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
import os
from dgl import save_graphs, load_graphs
from rdkit import Chem
from dgllife.utils import *
import dgl
from imblearn.over_sampling import RandomOverSampler

class Data_Process(DGLDataset):
    def __init__(self, path, name, save_dir, verbose=True):
        self.data = pd.read_csv(path)[['smiles', 'labels']]
        self.graph_molecules = []
        self.labels = []
        self.smiles=self.data['smiles']
        self.num_classes = 2
        super(Data_Process, self).__init__(
            name=name,
            verbose=verbose,
            save_dir=save_dir
        )

    def process(self):
        mol = [Chem.MolFromSmiles(x) for x in self.data['smiles']]
        node_featurizer=CanonicalAtomFeaturizer(atom_data_field='atomic')
        edge_featurizer = CanonicalBondFeaturizer(bond_data_field='atomic',self_loop=True)
        add_self_loop = True
        self.graph_molecules = [
            mol_to_graph(m, graph_constructor=partial(self.graph_constructor, add_self_loop=add_self_loop),
                         node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=True)
            for m in mol]
        self.labels=[x for x in self.data['labels']]
        return self.graph_molecules,self.labels


    def graph_constructor(self,mol,add_self_loop=True):
        # 创建无向图
        g = dgl.DGLGraph()
        g.add_nodes(mol.GetNumAtoms())
        # 添加原子之间的边
        bonds = mol.GetBonds()
        src, dst = [], []
        for bond in bonds:
            src.append(bond.GetBeginAtomIdx())
            dst.append(bond.GetEndAtomIdx())
        g.add_edges(src, dst)
        g.add_edges(dst, src)  # 添加反向边，构造无向图
        if add_self_loop:
            g = dgl.add_self_loop(g)  # 添加自环
        return g

    def save(self):
        graph_path = os.path.join(self.save_path, 'graph_data')
        save_graphs(graph_path, self.graph_molecules, {'labels': torch.tensor(self.labels)})

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph_data')
        self.graph_molecules, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'graph_data')
        return os.path.exists(graph_path)

    @staticmethod

    def featurize_edges(mol, add_self_loop=False):
        feats = []
        num_atoms = mol.GetNumAtoms()
        atoms = list(mol.GetAtoms())
        distance_matrix = Chem.GetDistanceMatrix(mol)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j or add_self_loop:
                    feats.append(float(distance_matrix[i, j]))
        return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

    @property
    def num_labels(self):
        return 2

    def __getitem__(self, idx):
        return self.graph_molecules[idx], self.labels[idx],self.smiles[idx]

    def __len__(self):
        return len(self.graph_molecules)


class Data_Sampling:
    def __init__(self,path):
        self.data=pd.read_csv(path)[['smiles', 'labels']]
        self.now_path=os.getcwd()

    def normsampling(self):
        train_file=self.now_path+"/data/train.csv"
        valid_file = self.now_path + "/data/valid.csv"
        test_file = self.now_path + "/data/test.csv"
        train_df, temp_df = train_test_split(self.data, test_size=0.3)
        valid_df, test_df = train_test_split(temp_df, test_size=0.66)
        train_df.to_csv(train_file, index=False)
        valid_df.to_csv(valid_file, index=False)
        test_df.to_csv(test_file, index=False)
        return train_file,valid_file,test_file

    def oversampling(self):
        train_file, valid_file, test_file=self.normsampling()
        data = pd.read_csv(train_file)
        X = data.drop('labels', axis=1)
        y = data['labels']  # 标签
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        train_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='labels')],axis=1)
        train_file="./data/train_oversampling.csv"
        train_df.to_csv(train_file, index=False)
        return train_file, valid_file, test_file

