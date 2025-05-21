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
