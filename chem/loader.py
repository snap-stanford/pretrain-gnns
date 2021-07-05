import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from tqdm import tqdm


elem_lst = None


def lookup_from_rdkit(elements):
    global elem_lst

    if elem_lst is None:
        print('calculating rdkit element representation lookup table')
        elem_rep_lookup = []
        for elem in elements:
            pt = Chem.GetPeriodicTable()

            if isinstance(elem, int):
                num = elem
                sym = pt.GetElementSymbol(num)
            else:
                num = pt.GetAtomicNumber(elem)
                sym = elem
            w = pt.GetAtomicWeight(elem)

            Rvdw = pt.GetRvdw(elem)
        #     Rcoval = pt.GetRCovalent(elem)
            valence = pt.GetDefaultValence(elem)
            outer_elec = pt.GetNOuterElecs(elem)

            elem_rep = [num, w,  Rvdw, valence, outer_elec]
#             print(elem_rep)

            elem_rep_lookup.append(elem_rep)
        elem_lst = elem_rep_lookup.copy()
        return elem_rep_lookup
    else:
        return elem_lst


def get_atom_rep(atomic_num, package='rdkit'):
    '''use rdkit or pymatgen to generate atom representation
    '''
    max_elem_num = 118
    element_nums = [x + 1 for x in range(max_elem_num)]

    if package == 'rdkit':
        elem_lst = lookup_from_rdkit(element_nums)
    elif package == 'pymatgen':
        raise Exception('pymatgen implementation is deprecated.')
        #elem_lst = lookup_from_pymatgen(element_nums)
    else:
        raise Exception('cannot generate atom representation lookup table')

    result = 0
    try:
        result = elem_lst[atomic_num - 1]
    except:
        print(f'error: atomic_num {atomic_num} does not exist')

    return result


def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph generation.')

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    atom_pos = []
    atom_attr = []

    # get atom attributes and positions
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        h = get_atom_rep(atomic_num)

        # , conf.GetAtomPosition(i).z])
        atom_pos.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        atom_attr.append(h)

    # get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, edge in enumerate(mol.GetBonds()):
        i = edge.GetBeginAtomIdx()
        j = edge.GetEndAtomIdx()

        bond_attr = None
        bond_type = edge.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_attr = [1]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_attr = [2]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_attr = [3]
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_attr = [4]

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)
#         print(f'i:{i} j:{j} bond_attr:{bond_attr}')

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)
#         print(f'j:{j} j:{i} bond_attr:{bond_attr}')

    x = torch.tensor(atom_attr)
    p = torch.tensor(atom_pos)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list)

    data = Data(x=x, p=p, edge_index=edge_index, edge_attr=edge_attr)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 D=2,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root
        self.D = D
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def get(self, idx):
    #     data = Data()
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(repeat(slice(None), item.dim()))
    #         s[data.__cat_dim__(key, item)] = slice(slices[idx],
    #                                                 slices[idx + 1])
    #         data[key] = item[s]
    #     return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return f'geometric_data_processed-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'adrb2_vae':
            smiles_lists = []
            data_list = []
            data_smiles_list = []

            for file, label, type_ in [('AID492947_active_T.smi', 1, 'train'),
                                       ('AID492947_active_V.smi', 1, 'val'),
                                       ('AID492947_inactive_T.smi', 0, 'train'),
                                       ('AID492947_inactive_V.smi', 0, 'val')]:
                smiles_path = os.path.join(self.root, 'raw', file)
                smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
                # labels = [label]* len(smiles_list)
                # types = [type_] * len(smiles_list)

                for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
                    smi = smiles_list[i]

                    data = smiles2graph(2, smi)
                    data.id = torch.tensor([i])
                    data.y = torch.tensor([label])
                    data.type = type_
                    # print(data)
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain


def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat(
            [torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset


def create_circular_fingerprint(mol, radius, size, chirality):
    """

    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)


class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
        processed_fp dir. The raw dir should contain the file containing the
        smiles, and the processed_fp dir can either be empty or a
        previously processed file
        :param dataset: name of dataset. Currently only implemented for
        tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information
        """
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality

        self._load()

    def _process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'chembl_with_labels':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(
                    os.path.join(self.root, 'raw'))
            print('processing')
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    fp_arr = create_circular_fingerprint(rdkit_mol,
                                                         self.radius,
                                                         self.size, self.chirality)
                    fp_arr = torch.tensor(fp_arr)
                    # manually add mol id
                    # id here is the index of the mol in
                    id = torch.tensor([i])
                    # the dataset
                    y = torch.tensor(labels[i, :])
                    # fold information
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y,
                                      'fold': fold})
                    data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(os.path.join(self.root, 'raw/tox21.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                     self.radius,
                                                     self.size,
                                                     self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor(labels[i, :])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(os.path.join(self.root, 'raw/HIV.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                     self.radius,
                                                     self.size,
                                                     self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor([labels[i]])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, 'processed_fp')
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False,
                                  header=False)
        with open(os.path.join(processed_dir,
                               'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(processed_dir,
                                   'fingerprint_data_processed.pkl'),
                      'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(
                self.root, self.dataset, self.radius, self.size, chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


# test MoleculeDataset object
if __name__ == "__main__":
    print('testing...')
    a = MoleculeDataset(
        D=2, root='D:/Documents/JupyterNotebook/Hit_Explosion/data/lit-pcba/VAE/ADRB2', dataset='adrb2_vae')
    print(a[0])

    data = a[0]
    print(data.x)
    print(data.p)
    print(data.edge_index)
    print(data.edge_attr)
    # create_all_datasets()
