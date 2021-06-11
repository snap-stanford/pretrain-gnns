import pymatgen.core as mg
import math
from rdkit import Chem
import torch

elements = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']


def lookup_from_pymatgen(elements):

    elem_lst = []
    for element in elements:
        if isinstance(element, int):
            elem = pt.Element.from_Z(element)
        else:
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
        #electron_affinity = elem.electron_affinity
        boiling_point = elem.boiling_point
        melting_point = elem.melting_point
        number = elem.number
        electronegativity = elem.X
        if math.isnan(electronegativity):
            electronegativity = 0
            print(f'{sym} has nan electronegativity')
#         print(float(elem.atomic_mass))
#         print(float(elem.atomic_radius))
#         print(float(elem.average_anionic_radius))
#         print(float(elem.average_cationic_radius))
#         print(float(elem.average_ionic_radius))
# #         print(float(elem.electron_affinity))
#         print(float(elem.boiling_point))
#         print(float(elem.melting_point))
#         print(float(elem.number))
#         print(float(elem.X))
        elem_rep = [mass, radius, average_anionic_radius, round(average_cationic_radius, 4), round(
            average_ionic_radius, 4), boiling_point, melting_point, number, electronegativity]
    


        elem_rep = [mass, radius, boiling_point, melting_point, number, electronegativity, ]

        # print(elem_rep)
        elem_lst.append(elem_rep)
    return elem_lst
    
def lookup_from_rdkit(elements):
    elem_rep_lookup = []
    for elem in elements:
        pt = Chem.GetPeriodicTable() 
        
        if isinstance(elem, int):
            num=elem
            sym=pt.GetElementSymbol(num)
        else:
            num = pt.GetAtomicNumber(elem)
            sym = elem
        w = pt.GetAtomicWeight(elem)
        
        Rvdw = pt.GetRvdw(elem)
    #     Rcoval = pt.GetRCovalent(elem)
        valence = pt.GetDefaultValence(elem)
        outer_elec = pt.GetNOuterElecs(elem)



        elem_rep=[w,  Rvdw, num, valence, outer_elec]
#         print(elem_rep)


        elem_rep_lookup.append(elem_rep)
    return elem_rep_lookup





def get_atom_rep(atomic_num, package):
	'''use rdkit or pymatgen to generate atom representation
	'''
	max_elem_num = 30
	element_nums= [x+1 for x in range(max_elem_num)]


	if package =='rdkit':
		elem_lst = lookup_from_rdkit(element_nums)
	elif package == 'pymatgen':
		elem_lst = lookup_from_pymatgen(element_nums)
	else:
		raise Exception('cannot generate atom representation lookup table')
	return elem_lst[atomic_num + 1]


def elem_rep_from_mol(mol):
    atom_lst = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_rep = get_atom_rep(atomic_num, 'rdkit')
        atom_lst.append(atom_rep)
    elem_rep = torch.tensor(atom_lst)
    return elem_rep


def trans_cis_augment(smi):
    if '=' not in smi:
        raise Exception(f'The molecule does not have double bond! smiles: {smi}')


def augment(smi):
    pass


if __name__ == '__main__':
    print('testing...')
    # smi1 = 'CC'
    # smi2 = 'C=C'
    # smi3 = 'C\C=CCF'
    # smi4 = 'C\C=C/CF'

    smi = 'CO'
    mol = Chem.MolFromSmiles(smi)
    print(elem_rep_from_mol(mol))
