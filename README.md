## **Usage**

### **1. GNN training**
To coarsen a graph (e.g., Cora dataset), run:

```bash
cd ./GNN
python graph_coarsening.py --dataname Cora --ratio 0.5
```
### **2. Training on a Coarsened Graph**
To train a model (e.g.GCN) on the coarsened graph:

```bash
python train.py --dataname Cora
```
### **2. PH computation**
1.download dataset 
Malnet-tiny：http://malnet.cc.gatech.edu/graph-data/

Oregon:https://snap.stanford.edu/data/Oregon-1.html

Enron:https://snap.stanford.edu/data/email-Enron.html

P2P:https://snap.stanford.edu/data/p2p-Gnutella31.html


2.Unzip dataset in PH_data/{dataname}

3.Run experiment

```bash
python ripser_experiment.py --dataname {dataname}
```
