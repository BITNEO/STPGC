## **Usage**

### **1. GNN training**
To coarsen a graph (e.g., Cora dataset), run:

```bash
cd ./GNN
python graph_coarsening.py --dataname Cora --ratio 0.5
```

To train a model (e.g.GCN) on the coarsened graph:

```bash
python train.py --dataname Cora
```
### **2. PH computation**
1.Download dataset 

REDDIT5K: https://chrsmrrs.github.io/datasets/docs/home/

Oregon:https://snap.stanford.edu/data/Oregon-1.html

Enron:https://snap.stanford.edu/data/email-Enron.html

PDB:https://drive.google.com/file/d/1UiM_lD9KkTJRvu5sogKUgIdHakR2sNZI/view?usp=drive_link

P2P:https://snap.stanford.edu/data/p2p-Gnutella31.html


2.Unzip dataset in PH_data/{dataname}

3.Run experiment

```bash
python ripser_experiment.py --dataname {dataname}
```
