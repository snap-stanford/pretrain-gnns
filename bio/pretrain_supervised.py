import argparse

from splitters import random_split, species_split
from loader import BioDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd

from util import combine_dataset

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.go_target_pretrain.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()

        loss_accum += loss.detach().cpu()

    return loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default = "species", help='Random or species split')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    root_supervised = 'dataset/supervised'

    dataset = BioDataset(root_supervised, data_type='supervised')

    if args.split == "random":
        print("random splitting")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, seed = args.seed)
        print(train_dataset)
        print(valid_dataset)
        pretrain_dataset = combine_dataset(train_dataset, valid_dataset)
        print(pretrain_dataset)
    elif args.split == "species":
        print("species splitting")
        trainval_dataset, test_dataset = species_split(dataset)
        test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5, frac_valid=0.5, frac_test=0)
        print(trainval_dataset)
        print(test_dataset_broad)
        pretrain_dataset = combine_dataset(trainval_dataset, test_dataset_broad)            
        print(pretrain_dataset)
        #train_dataset, valid_dataset, _ = random_split(trainval_dataset, seed = args.seed, frac_train=0.85, frac_valid=0.15, frac_test=0)
    else:
        raise ValueError("Unknown split name.")


    train_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    num_tasks = len(pretrain_dataset[0].go_target_pretrain)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)   
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model, device, train_loader, optimizer)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")



if __name__ == "__main__":
    main()
