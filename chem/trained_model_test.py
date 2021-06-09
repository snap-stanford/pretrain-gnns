import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import loader

import os
import shutil

from tensorboardX import SummaryWriter
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch


num_layer = 5
emb_dim =300
num_tasks = 27
JK = 'last'
dropout_ratio = 0
graph_pooling = 'mean'
gnn_type = 'gin'


def canonicalize(smi):
	smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
	return smi

# set up model
model = GNN_graphpred(num_layer, emb_dim, num_tasks, JK=JK,
                      drop_ratio=dropout_ratio, graph_pooling=graph_pooling, gnn_type=gnn_type)




model.load_state_dict(torch.load('output/raw_model.pth'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()




smi1 = 'C(=C(\Cl)C)(/C)Cl'#identical cis molecule
smi2 = 'C(=C(/Cl)C)(\C)Cl'#

smi1 = '[C@](C)(O)([F])[Cl]' #identical chiral molecule
smi2 = 'C[C@@]([F])([Cl])O'#'[C@]([Cl])([F])(O)C' 

#smi1 = canonicalize(smi1)
#smi2 = canonicalize(smi2)

# smi1 = 'C(=C)C'#identical molecule without cis-trans isomerism
# smi2 = 'CC=C'#


#smi1 = 'C(=C(\Cl)C)(/C)Cl'#cis
#smi2 = 'C(=C(/Cl)C)(/C)Cl' #trans

data1 = loader.mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smi1))
data2 = loader.mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smi2))

dataset = [data1, data2]

loader = DataLoader(dataset, batch_size = 1)


result_h_list = []
for batch in loader:
	batch.to(device)
		
	pred, h = model(batch.x, batch.edge_index,
	                     batch.edge_attr, batch.batch)
	print(f'shape:{(h.shape)} representation:{h}')
	result_h_list.append(h)

diff = result_h_list[1]-result_h_list[0]
print(torch.where(diff>0.001, 1, 0))
