## STPGC: Strong Topology-Preserving Graph Coarsening

This repository implements the STPGC method for graph coarsening that preserves topological structures of the original graph, along with GCN training on the coarsened graph.




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




