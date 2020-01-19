import torch
import random
import numpy as np

def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                 seed=0):
    """
    Adapted from graph-pretrain
    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :return: train, valid, test slices of the input dataset obj.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    if frac_test == 0:
        test_dataset = None
    else:
        test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def species_split(dataset, train_valid_species_id_list=[3702, 6239, 511145,
                                                        7227, 10090, 4932, 7955],
                  test_species_id_list=[9606]):
    """
    Split dataset based on species_id attribute
    :param dataset:
    :param train_valid_species_id_list:
    :param test_species_id_list:
    :return: train_valid dataset, test dataset
    """
    # NB: pytorch geometric dataset object can be indexed using slices or
    # byte tensors. We will use byte tensors here

    train_valid_byte_tensor = torch.zeros(len(dataset), dtype=torch.uint8)
    for id in train_valid_species_id_list:
        train_valid_byte_tensor += (dataset.data.species_id == id)

    test_species_byte_tensor = torch.zeros(len(dataset), dtype=torch.uint8)
    for id in test_species_id_list:
        test_species_byte_tensor += (dataset.data.species_id == id)

    assert ((train_valid_byte_tensor + test_species_byte_tensor) == 1).all()

    train_valid_dataset = dataset[train_valid_byte_tensor]
    test_valid_dataset = dataset[test_species_byte_tensor]

    return train_valid_dataset, test_valid_dataset

if __name__ == "__main__":
    from collections import Counter
