import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from clearml import Task

from model import MolGCN, GNN_graphpred
from evaluation import enrichment
from util import print_model_size

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()


def train(args, model, device, loader, optimizer):
    model.train()

    loss_lst = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch = batch.to(device)
        # batch.x = batch.x.to(device)
        # batch.p = batch.p.to(device)
        # batch.edge_index = batch.edge_index.to(device)
        # batch.edge_attr = batch.edge_attr.to(device)
        batch.batch = batch.batch.to(device)

        batch = batch.to(device)

        pred, h = model(batch.x, batch.p, batch.edge_index,
                        batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        # is_valid = y**2 > 0
        # Loss matrix
        # print(f"pred:{pred.double()}, y:{y}")
        loss = criterion(pred.double(), y)
        # loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(
        #     loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        # print(f'loss:{loss}')
        loss_lst.append(loss)
        optimizer.zero_grad()
        # loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        torch.cuda.empty_cache()

        optimizer.step()
    batch_loss = sum(loss_lst) / float(len(loader))
    print(f'loss:{batch_loss}')


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred, h = model(batch.x, batch.p, batch.edge_index,
                            batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        print(f'batch.y:{batch.y}')

        print(f'pred:{pred}')
        pred = torch.where(pred > 0.5, torch.ones(
            pred.shape, device=device), torch.zeros(pred.shape, device=device))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []

    for i in range(y_true.shape[1]):
        roc_list.append(enrichment(y_true[:, i], y_scores[:, i]))
    # for i in range(y_true.shape[1]):
    #     # AUC is only defined when there is at least one positive data.
    #     if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
    #         is_valid = y_true[:, i]**2 > 0
    #         roc_list.append(roc_auc_score(
    #             (y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %
              (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def main():
    # start clearml as the task manager. this is optional.

    # task = Task.init(project_name="kernel GNN", task_name="out-of-memory")

    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str,
                        default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold",
                        help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == "adrb2":
        num_tasks = 1
    elif args.dataset == '435008':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    # windows
    # root = 'D:/Documents/JupyterNotebook/GCN_property/pretrain-gnns/chem/dataset/'
    # linux
    root = '~/projects/GCN_Syn/examples/pretrain-gnns/chem/dataset/'
    if args.dataset == '435008':
        root = root + 'qsar_benchmark2015'
        dataset = args.dataset
    else:
        raise Exception('cannot find dataset')

    dataset = MoleculeDataset(D=2, root=root, dataset=dataset)
    index = list(range(400)) + list(range(1000, 1400))
    dataset = dataset[index]
    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv(
            "D:/Documents/JupyterNotebook/Hit_Explosion/data/lit-pcba/VAE/" + args.dataset.upper() + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(
            'dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random scaffold")
    elif args.split == 'lit-pcba':
        # train_dataset
        pass
    else:
        raise ValueError("Invalid split option.")

    print(
        f'training size: {len(train_dataset)}, actives: {len(torch.nonzero(torch.tensor([data.y for data in train_dataset])))}')
    print(
        f'valid size: {len(valid_dataset)}, actives: {len(torch.nonzero(torch.tensor([data.y for data in valid_dataset])))}')
    print(
        f'test size: {len(test_dataset)}, actives: {len(torch.nonzero(torch.tensor([data.y for data in test_dataset])))}')

    print('loading data')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset),
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=len(
        test_dataset), shuffle=False, num_workers=args.num_workers)
    print('data loaded!')
    # set up model
    model = GNN_graphpred(num_layers=4, num_kernel_layers=15, x_dim=5, p_dim=3, edge_attr_dim=1, num_tasks=num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
    # check model size
    print_model_size(model)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model = model.to(device)
    # print(f'model device cuda:{next(model.parameters()).is_cuda}')

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + \
                str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()
    torch.save(model.state_dict(), "output/trained_model.pth")


if __name__ == "__main__":
    main()
