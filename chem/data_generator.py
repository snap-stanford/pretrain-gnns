from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric import InMemoryDataset


# elements = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
# bonds = ['-', '=', '#']


# def is_valid(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#     except:
#         print(f'error:{smiles}')

#     if mol is not None:
#         # rdmolops.AddHs(mol)
#         try:
#             Chem.SanitizeMol(mol)
#             return True
#         except:
#             print(f'{smiles} is not chemically valid')
#             return False
#     return False


# def add_atom(input_smiles):
#     new_list = []
#     print(f'input:{input_smiles}')
#     for elem in elements:
#         for bond in bonds:
#             smiles = input_smiles
#             smiles += bond
#             smiles += elem

#             print(f'generated:{smiles}')
#             if is_valid(smiles):
#                 print(f'validated:{smiles}')
#                 new_list.append(smiles)

#     return new_list


# def generate_molecule_list(num_atoms=2):

#     molecule_lst = []
#     generation_seeds = elements
#     temp_seeds = []

#     # for num in range(num_atoms - 1):
#     for smiles in generation_seeds:

#         # print(f'before adding:{smiles}')

#         smiles_list = add_atom(smiles)
#         molecule_lst += smiles_list

#     generation_seeds = molecule_lst.copy()

#     print(f'generation_seeds:{generation_seeds}')
#     for smiles in generation_seeds:

#         #     # print(f'before adding:{smiles}')

#         smiles_list = add_atom(smiles)
#     molecule_lst += smiles_list

#     return molecule_lst

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
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
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []


if __name__ == '__main__':
    print('testing')
    dataset = MoleculeDataset()
    # lst = generate_molecule_list(3)
    # print(f'data size:{len(lst)}')
    # for smi in lst:
    #     print(smi)
    #     if is_valid(smi) == False:
    #         print(f'{smi} is not valid')
