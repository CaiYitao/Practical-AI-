from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from torch_geometric.data import (InMemoryDataset,Dataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph
from e3nn.o3 import Irreps, spherical_harmonics
import pickle
import pandas as pd


class TOX21(InMemoryDataset):
    def __init__(self, root, partition, lmax_attr,fold_idx=0,transform=None, pre_transform=None):
    
        assert partition in ["train", "test", "valid","CVtrain","CVtest"]
        self.root = root
  
        self.partition = partition
        self.fold_idx = fold_idx
 
        self.lmax_attr = lmax_attr
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)


        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["~/confdata.pkl","~/tox21.csv"]


    @ property
    def processed_file_names(self) -> str:
        return ["_".join([self.partition,  "l="+str(self.lmax_attr)]) + '.pt']   
    
    def download(self):
        pass

    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            print("Please install rdkit")
            return

        with open(self.raw_paths[1], 'rb') as f:
            tox21 = pd.read_csv(f)
            target = tox21.iloc[:,8:].values
            indx = tox21["CVfold"].values
            indx = indx[0:11704]  
            #map nan to 0, 0 to -1, 1 to 1
            for i in range(len(target)):
                for j in range(len(target[i])):
                    if np.isnan(target[i][j]):
                        target[i][j] = 0
                    elif target[i][j] == 0:
                        target[i][j] = -1
                    elif target[i][j] == 1:
                        target[i][j] = 1
            target = torch.tensor(target, dtype=torch.float)


        with open(self.raw_paths[0], 'rb') as f:
            conf = pickle.load(f)

        types = {'H': 0,  'C': 1,  'N': 2,  'O': 3,  'F': 4,  'P': 5,  'S': 6,  'Ag': 7,  'Al': 8,  'As': 9, 
         'Au': 10,  'B': 11,  'Ba': 12,  'Be': 13,  'Bi': 14,  'Br': 15,  'Ca': 16,  'Cd': 17,  'Cl': 18,
         'Co': 19,  'Cr': 20,  'Cu': 21,  'Dy': 22,  'Eu': 23,  'Fe': 24,  'Gd': 25,  'Ge': 26,  'Hg': 27,  
         'I': 28,  'In': 29,  'K': 30,  'Li': 31,  'Mg': 32,  'Mn': 33,  'Mo': 34,  'Na': 35,  'Nd': 36,  
         'Ni': 37,  'Pb': 38,  'Pd': 39,  'Pt': 40,  'Ru': 41,  'Sb': 42,  'Sc': 43,  'Se': 44,  'Si': 45,  
         'Sn': 46,  'Ti': 47,  'Tl': 48,  'V': 49,  'Yb': 50,  'Zn': 51,  'Zr': 52}

        data_list = []
        train, valid, test = dict(list(conf.items())[0:11704]), dict(list(conf.items())[11704:11999]), dict(list(conf.items())[11999:])
        traintarget, validtarget, testtarget = target[0:11704,:], target[11704:11999],target[11999:]
        CVtrain,CVtest,CVtrain_target,CVtest_target = [],[],[],[]
        if self.partition == "CVtrain" or self.partition == "CVtest":
            for i,idx in enumerate(indx):
             
                if idx == self.fold_idx:
                    CVtest.append(list(train.items())[i])
                    CVtest_target.append(traintarget[i].tolist())
                else:
                    CVtrain.append(list(train.items())[i])
                    CVtrain_target.append(traintarget[i].tolist())
        
        CVtrain,CVtest,CVtrain_target,CVtest_target = dict(CVtrain),dict(CVtest),torch.tensor(CVtrain_target,dtype=torch.float),torch.tensor(CVtest_target,dtype=torch.float)


        indices = {"train": train, "valid": valid, "test": test, "CVtrain": CVtrain, "CVtest": CVtest}
        targetindices = {"train": traintarget, "valid": validtarget, "test": testtarget, "CVtrain": CVtrain_target, "CVtest": CVtest_target}
    
        Nmols = len(conf)
        np.random.seed(0)

   
        for i,((name,confdata),target) in enumerate(zip(indices[self.partition].items(), targetindices[self.partition])):
         
            mol = confdata.rdmol
            N = mol.GetNumAtoms()            
            pos = confdata.pos
            edge_index = confdata.edge_index

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
        
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            
            z = torch.tensor(atomic_number, dtype=torch.long)       
            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)
            y = target.unsqueeze(0)
       
            name = name

            edge_attr, node_attr, edge_dist = self.get_O3_attr(edge_index, pos, self.attr_irreps)

            data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                        node_attr=node_attr, additional_message_features=edge_dist, y=y, name=name, index=i)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
    
    def get_O3_attr(self, edge_index, pos, attr_irreps):
        """ Creates spherical harmonic edge attributes and node attributes for the SEGNN """
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
        edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
        edge_attr = spherical_harmonics(attr_irreps, rel_pos, normalize=True,
                                        normalization='component')  # Unnormalised for now
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean")
        return edge_attr, node_attr, edge_dist




