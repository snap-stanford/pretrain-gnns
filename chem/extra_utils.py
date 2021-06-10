import pymatgen.core as mg
import math
from rdkit import Chem
import torch

elements = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
elem_lst = []
for element in elements:
    elem = mg.Element(element)

    sym = elem.symbol
    mass = elem.atomic_mass
    radius = elem.atomic_radius
    if radius is None:
        radius = 1
        print(f'{sym} has None radius')
    average_anionic_radius = elem.average_anionic_radius
    average_cationic_radius = elem.average_cationic_radius
    average_ionic_radius = elem.average_ionic_radius
    electron_affinity = elem.electron_affinity
    boiling_point = elem.boiling_point
    melting_point = elem.melting_point
    number = elem.number
    electronegativity = elem.X
    if math.isnan(electronegativity):
        electronegativity = 0
        print(f'{sym} has nan electronegativity')
#     print(float(elem.atomic_mass))
#     print(float(elem.atomic_radius))
#     print(float(elem.average_anionic_radius))
#     print(float(elem.average_cationic_radius))
#     print(float(elem.average_ionic_radius))
#     print(float(elem.electron_affinity))
#     print(float(elem.boiling_point))
#     print(float(elem.melting_point))
#     print(float(elem.number))
#     print(float(elem.X))
    elem_rep = [mass, radius, average_anionic_radius, round(average_cationic_radius, 4), round(
        average_ionic_radius, 4), electron_affinity, boiling_point, melting_point, number, electronegativity]
    elem_lst.append(elem_rep)


def get_atom_rep(atomic_num):
    return elem_lst[atomic_num + 1]


def X_from_mol(mol):
    atom_lst = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_rep = get_atom_rep(atomic_num)
        atom_lst.append(atom_rep)
    X = torch.tensor(atom_lst)
    return X


def trans_cis_augment(smi):
    if '=' not in smi:
        raise f'The molecule does not have double bond! smiles: {smi}'


def augment(smi):
    pass


if __name__ == '__main__':
    print('testing...')
    # smi1 = 'CC'
    # smi2 = 'C=C'
    # smi3 = 'C\C=CCF'
    # smi4 = 'C\C=C/CF'

    smi = 'CC'
    mol = Chem.MolFromSmiles(smi)
    print(X_from_mol(mol))
