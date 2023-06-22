import argparse

from loader import BioDataset
from dataloader import DataLoaderMasking 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred

import pandas as pd

from util import MaskEdge

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

def compute_accuracy(pred, target):
    #return float(torch.sum((pred.detach() > 0) == target.to(torch.uint8)).cpu().item())/(pred.shape[0]*pred.shape[1])
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_edges = model_list
    optimizer_model, optimizer_linear_pred_edges = optimizer_list

    model.train()
    linear_pred_edges.train()

    loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        ### predict the edge types.
        masked_edge_index = batch.edge_index[:, batch.masked_edge_idx]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear_pred_edges(edge_rep)

        #converting the binary classification to multiclass classification
        edge_label = torch.argmax(batch.mask_edge_label, dim = 1)

        acc_edge = compute_accuracy(pred_edge, edge_label)
        acc_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()

        loss = criterion(pred_edge, edge_label)
        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_edges.step()

        loss_accum += float(loss.cpu().item())

    return loss_accum/(step + 1), acc_accum/(step + 1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
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
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f" %(args.num_layer, args.mask_rate))

    #set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform = MaskEdge(mask_rate = args.mask_rate))

    print(dataset)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)


    #set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    #Linear layer for classifying different edge types
    linear_pred_edges = torch.nn.Linear(args.emb_dim, 7).to(device)

    model_list = [model, linear_pred_edges]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_edges = optim.Adam(linear_pred_edges.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_edges]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc)

    if not args.model_file == "":
        torch.save(model.state_dict(), args.model_file + ".pth")


if __name__ == "__main__":
    main()
