
This repository is the official code implementation for the paper:

**Scalable Topology-Preserving Graph Coarsening: Concepts and Algorithms**
*ICML 2026*





### Usage

#### 1. Graph Coarsening

Coarsen a graph (e.g., Cora dataset with 50% reduction ratio):

```bash
cd ./GNN
python graph_coarsening.py --dataname Cora --ratio 0.5
```

Coarsened graphs are saved to `coarsened_graph/`.

#### 2. GCN Training on Coarsened Graph

Train a GCN model on the coarsened graph:

```bash
cd ./GNN
python train_gcn.py --dataset Cora --coarsening_ratio 0.5 --runs 10
```

Key arguments:
- `--dataset`: Dataset name (Cora, Citeseer, pubmed, dblp, etc.)
- `--coarsening_ratio`: Graph reduction ratio (default: 0.3)
- `--runs`: Number of training runs (default: 10)
- `--epochs`: Training epochs (default: 200)
- `--hidden`: Hidden layer size (default: 256)
- `--lr`: Learning rate (default: 0.01)


If you find this repository useful, please consider citing:

```bibtex
@inproceedings{wu2026stpgc,
  title={Scalable Topology-Preserving Graph Coarsening: Concepts and Algorithms},
  author={Xiang Wu, Rong-Hua Li, Xunkai Li, Kangfei Zhao,
  Hongchao Qin, and Guoren Wang},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

