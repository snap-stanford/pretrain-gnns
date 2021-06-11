from rdkit import Chem
from rdkit.Chem import rdmolops



elements = ['H','C','N', 'O', 'P', 'S', 'F','Cl', 'Br', 'I']
bonds = ['-', '=', '#']


def generate_2_atoms():
	molecule_lst = []
	
	
	for elem in elements:
		for smi in elements:
			for bond in bonds:
				smiles = smi
				smiles += bond
				smiles += elem

				mol = Chem.MolFromSmiles(smiles)
				if  mol is not None:
					# rdmolops.AddHs(mol)
					try:
						Chem.SanitizeMol(mol)
						molecule_lst.append(smiles)
					except:
						print(f'{smiles} is not chemically valid')

	return  molecule_lst


if __name__ == '__main__':
	lst = generate_2_atoms()
	print(f'data size:{len(lst)}')
	# for smi in lst:
	# 	print(f'{smi}, valid:{Chem.MolFromSmiles(smi) is not None}')

