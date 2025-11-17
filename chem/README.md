# Chemistry GNNs — Reproduction of "Strategies for Pre-training Graph Neural Networks"

This directory contains a cleaned-up, minimally maintained reproduction of the chemistry part of:

Weihua Hu*, Bowen Liu*, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. "Strategies for Pre-training Graph Neural Networks." ICLR 2020.

Some legacy code paths were simplified to rely more directly on PyTorch Geometric (PyG). A few pragmatic fixes were added for newer PyTorch/PyG/RDKit versions while keeping the original design intact.

## Why this folder?

The `chem/` folder provides:

- Data utilities to convert SMILES to PyG graphs (`data.py`)
- A flexible GNN backbone and graph-level prediction head (`model.py`)
- Self-supervised context prediction pretraining (`pretrain_contextpred.py`)
- Supervised fine-tuning for classification tasks (`finetune.py`)
- Common dataset splitters (scaffold, random, random_scaffold) (`splitters.py`)

Run commands from the repository root so relative paths like `dataset/...` resolve correctly.

## Installation

Tested with Python 3.11. Create a virtual environment and install dependencies from the repo root:

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- You need a working PyTorch + PyTorch Geometric setup. If installation fails, follow the official PyG installation instructions for your CUDA/macOS setup, then re-run `pip install -r requirements.txt`.
- RDKit is required for SMILES parsing.

## Directory structure (chem/)

- `data.py`: SMILES -> PyG `Data` conversion, dataset readers (HIV, BACE, JCER), caching to `processed/data.pt`.
- `model.py`: backbone GNNs (GIN/GCN/GraphSAGE/GAT) and `GNN_graphpred` for graph-level classification.
- `splitters.py`: scaffold split, random split, and random_scaffold split utilities.
- `pretrain_contextpred.py`: self-supervised pretraining via substructure-context prediction (CBOW/skip-gram style).
- `finetune.py`: supervised fine-tuning with ROC-AUC metrics, optional early stopping and best-checkpoint saving.

## Data preparation

The code expects dataset directories under `dataset/` (run from repo root):

```
dataset/
        hiv/
            raw/
                hiv.csv
            processed/
                ... (auto-generated)
        bace/
            raw/
                bace.csv
        jcesr/
            raw/
                jcesr.csv
        zinc_standard_agent/
            raw/
                zinc_standard_agent.csv.gz  # for self-supervised pretraining example
```

Each dataset has its own expected columns:

- HIV (`dataset=hiv`): CSV with columns `smiles`, `HIV_active`. Labels will be mapped from 0 -> -1. Final labels should be in {−1, 1}, and 0 means "missing" (ignored by the loss).
- BACE (`dataset=bace`): CSV with columns `mol`, `Class`, `Model` where `Model` is one of Train/Valid/Test. 0s in `Class` are mapped to −1. The code also stores `fold` from `Model`.
- JCER (`dataset=jcesr`): CSV with columns `smiles`, `Class`. 0s in `Class` are mapped to −1.
- ZINC standard agent (`dataset=zinc_standard_agent`): gzipped CSV with `smiles` and `zinc_id`, used for context-prediction pretraining.

Processed artifacts are cached under `dataset/<name>/processed/`, including `smiles.csv` and `data.pt`.

## How tasks are defined (num_tasks)

In `finetune.py`, `num_tasks` is the number of binary classification targets (multi-task setup). It determines the output dimension of the prediction head in `GNN_graphpred`. The script currently sets `num_tasks` per known dataset name, e.g. `tox21=12`, `pcba=128`, `toxcast=617`, while datasets like `hiv`, `bace`, `bbbp`, `jcesr` are single-task (`num_tasks=1`). Labels are expected in {−1, 1}, with 0 treated as missing and masked out during loss/AUC.

Tip: if you add a new dataset, ensure the number of label columns matches `num_tasks`.

## Splitting strategies

Choose via `--split` in `finetune.py`:

- `scaffold`: Bemis–Murcko scaffold split (deterministic)
- `random`: random split
- `random_scaffold`: scaffold split with random scaffold ordering

`splitters.py` contains implementations and docstrings.

## Self-supervised pretraining (context prediction)

Example (CBOW mode on ZINC standard agent):

```bash
# from repo root
python chem/pretrain_contextpred.py \
    --dataset zinc_standard_agent \
    --output_model_file runs/pretrained/contextpred_gnn \
    --num_layer 5 --emb_dim 300 --dropout_ratio 0.0 --gnn_type gin \
    --epochs 100 --batch_size 256 --lr 1e-3
```

Key arguments:
- `--mode {cbow,skipgram}`: prediction objective
- `--csize`: context radius (used to derive subgraph sizes)
- `--num_layer`, `--emb_dim`, `--JK`, `--dropout_ratio`, `--gnn_type`: GNN config
- `--dataset`: dataset folder name under `dataset/`
- `--output_model_file`: path prefix for saving the backbone weights (`.pth` appended)

The saved weights contain the GNN backbone only; the classification head will be initialized during fine-tuning.

## Supervised fine-tuning

Example (HIV, single task):

```bash
# from repo root
python chem/finetune.py \
    --dataset hiv \
    --input_model_file runs/pretrained/contextpred_gnn.pth \
    --filename runs/finetune/hiv \
    --num_layer 5 --emb_dim 300 --dropout_ratio 0.5 --graph_pooling mean \
    --epochs 100 --batch_size 32 --lr 1e-3 --split scaffold \
    --save_best 1 --early_stop 0
```

What it does:
- Loads `dataset/hiv`, performs the chosen split, and builds loaders
- Initializes `GNN_graphpred(num_tasks=1)` and optionally loads a pretrained backbone via `--input_model_file`
- Trains with `BCEWithLogitsLoss` per task and reports ROC-AUC on train/val/test
- If `--filename` is set, logs TensorBoard scalars under `runs/finetune_cls_runseed<seed>/<filename>` and saves the best checkpoint as `<filename>_best.pth` when `--save_best=1`

Important arguments in `finetune.py`:
- Data/split: `--dataset`, `--split {scaffold,random,random_scaffold}`, `--seed` (split seed), `--num_workers`
- Model: `--num_layer`, `--emb_dim`, `--JK {last,sum,max,concat}`, `--dropout_ratio`, `--graph_pooling {sum,mean,max,set2set,attention}`, `--gnn_type {gin,gcn,graphsage,gat}`
- Optimization: `--epochs`, `--batch_size`, `--lr`, `--lr_scale` (head LR multiplier), `--decay` (weight decay)
- Repro/checks: `--runseed` (init seed), `--shuffle_labels` (sanity check; expect AUC≈0.5), `--eval_train`
- Training mgmt: `--save_best`, `--early_stop`, `--patience`

## Adding a new dataset (example: JCER)

If your CSV schema is `smiles` + a single binary label column (0/1), you can plug it in with minimal code. The loader in `data.py` contains `_load_jcesr_dataset(...)` and a corresponding branch in `MoleculeDataset.process`. To add a similar dataset:

1) Prepare the CSV under `dataset/<your_dataset>/raw/your_dataset.csv` with at least:

- `smiles`: SMILES string
- `Class`: binary label; 0 will be converted to −1 internally

2) Call fine-tuning:

```bash
# from repo root
python chem/finetune.py --dataset jcesr --split scaffold --epochs 100 --batch_size 32
```

3) If you add multi-task labels, ensure `finetune.py` sets the correct `num_tasks` for your dataset name and that your CSV provides the matching number of label columns. Alternatively, refactor to infer `num_tasks` from the processed data shape.

## Tips and troubleshooting

- Run from the `pre/` repo root so relative paths like `dataset/<name>` resolve.
- Invalid SMILES are skipped safely; if you see `No valid molecules were found...`, check your CSV.
- Labels should be in {−1, 1}. Any 0s are treated as missing and masked out during loss/AUC.
- For TensorBoard logs, point your viewer to the `runs/` directory.
- On macOS/CPU-only setups, PyG installation may require specific wheels; consult the PyG docs.

## Citation

If you use this code, please cite the original paper:

Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for Pre-training Graph Neural Networks. ICLR 2020.

## License

This is a research reproduction with modifications. Please refer to the upstream project and this repository's license for details.
