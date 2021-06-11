from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from torch_geometric.data import Data

from extra_utils import get_atom_rep
from model import GNN

allowable_features = {
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]


    }



def train(model_list, optimizer_list):
	model, pred_vol = model_list
	optimizer, pred_vol_optimizer = optimizer_list
	
	model.train(model, device)

	for step, batch in enumerate(tqdm(loader, desc="Iteration")):
	
		batch = batch.to(device)
		node_rep = model(batch.x, batch.edge_index, batch.edge_attr)


def main():
	num_layer = 2
	num_dim =300
	JK = 'last'
	drop_ratio = 0
	gnn_type = 'GIN'

	device = 'gpu' if torch.cuda.is_available() else 'cpu'

	model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)


def graph_from_mol(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms

    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        h = get_atom_rep(atom.GetAtomicNum(), 'rdkit')
        # print(h)
        atom_feature = get_atom_rep(atom.GetAtomicNum(), 'rdkit') + [allowable_features[
             'possible_chirality_list'].index(atom.GetChiralTag())]

        # atom_feature = [allowable_features['possible_atomic_num_list'].index(
        #     atom.GetAtomicNum())] + [allowable_features[
        #     'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)



    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    AllChem.EmbedMolecule(mol)
    volume = AllChem.ComputeMolVolume(mol)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, volume = volume)

    return data




if __name__ =='__main__':
	print('testing...')
	smi = 'CO'
	print(f'smi:{smi}')
	mol = Chem.MolFromSmiles(smi)



	data = graph_from_mol(mol)
	print(data)



