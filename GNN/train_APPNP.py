import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net,GCN,APPNP_Net
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull,Coauthor
from PyGdataset import PygNodePropPredDataset,Evaluator
import numpy as np
from torch_geometric.utils import to_undirected,add_self_loops
import os
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
#Citeseer:lr = 0.001 Cora:0.1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    for dataname in ['ogbn-products']:
        for deg1 in [60,70,80]:
            # for v in [0.3, 0.5, 0.7, 0.8, 0.9]:
            for v in [0.1]:
                
            # for v in [0.8, 0.9]:
            # for v in [ 0.9]:
                #deg1 = 50
                deg2 = deg1
                #data_mol, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_offline.npy' % (dataname, v, rank), allow_pickle=True)
                #data_mol,m  = np.load('/home/wuxiang/GEC-main/Reduced_Node_Data/%s_%.2f_split%d_All_Simplex_1.npy' % (dataname, v, rank), allow_pickle=True)
                #data_mol, m = np.load('/home/wuxiang/strong_collapse/Reduced_Node_Data/%s_%.2f_split_All_Simplex_1.npy' % (dataname, v), allow_pickle=True)
                data_mol, m = np.load('/home/wuxiang/strong_collapse/hyperparameter_study/%s_%.2f_%d_%d_%dsplit_All_Simplex_3.npy' % (dataname, v,0,deg1,deg2), allow_pickle=True)
                # data_mol, m = np.load('/home/ycmeng/ep2/Reduced_Node_Data/%s_%.2f_split%d_link_7.npy' % (dataname, v, rank), allow_pickle=True)
                #print(data_mol)
                if dataname in ['Cora','Citeseer','pubmed']:
                    dataset = Planetoid(root='/home/wuxiang/GEC-main/', name=dataname)
                    data = dataset[0]
                elif dataname in ['dblp']:
                    dataset = CitationFull(root='/home/wuxiang/GEC-main/dataset', name=dataname)
                    data = dataset[0]
                elif dataname in ['Physics']:
                    dataset = Coauthor(root='/home/wuxiang/GEC-main/dataset', name=dataname)
                    data = dataset[0]
                    edges = data.edge_index
                    label = data.y
                    

                elif dataname == "ogbn-arxiv":
                    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/wuxiang/strong_collapse/dataset/arxiv')
                    split_idx = dataset.get_idx_split()
                    evaluator = Evaluator('ogbn-arxiv')
                    data = dataset[0]
                    data.edge_index = to_undirected(data.edge_index)
                    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
                    split_idx = dataset.get_idx_split()
                    data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
                    data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
                    data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
                    data.y= data.y.view(-1)
                    label = data.y

                elif dataname == "ogbn-products":
                    dataset = PygNodePropPredDataset(name='ogbn-products', root='/mnt/ssd2/products/raw')
                    split_idx = dataset.get_idx_split()
                    evaluator = Evaluator('ogbn-products')
                    data = dataset[0]
                    #data.adj_t = data.adj_t.to_symmetric()
                    data.edge_index = to_undirected(data.edge_index)
                    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
                    split_idx = dataset.get_idx_split()
                    data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
                    data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
                    data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
                    data.y= data.y.view(-1)
                    label = data.y
                    
                else:
                    dataset = Coauthor(root='/home/wuxiang/GEC-main/dataset', name=dataname)
                
                if dataname == 'dblp' or dataname == 'Physics':
                    indices = []
                    num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
                    for i in range(num_classes):
                        index = (data.y == i).nonzero().view(-1)
                        index = index[torch.randperm(index.size(0))]
                        indices.append(index)

                    train_index = torch.cat([i[:int(len(i)*0.7)] for i in indices], dim=0)
                    val_index = torch.cat([i[int(len(i)*0.7):int(len(i)*0.8)] for i in indices], dim=0)
                    test_index = torch.cat([i[int(len(i)*0.8):] for i in indices], dim=0)
                    print(data.num_nodes)
                    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
                    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
                    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
                
                data_mol.x = data.x
                data_mol.y = data.y
                data_mol.edge_index = data.edge_index
                data_mol.train_mask = data.train_mask
                data_mol.val_mask = data.val_mask
                data_mol.test_mask = data.test_mask
                # l1 = []
                # l2 = []
                # edges = {}
                # for edge_num in range(data.edge_index.size()[1]):
                #     if (int(data.edge_index[0][edge_num]) in m.keys()) and (int(data.edge_index[1][edge_num]) in m.keys()):
                #         node1 = m[int(data.edge_index[0][edge_num])] 
                #         node2 = m[int(data.edge_index[1][edge_num])]
                #         if node1 == node2:
                #             continue
                #         if (min(node1, node2), max(node1, node2)) in edges.keys():
                #             # print("exist edge")
                #             continue
                #         l1.append(node1)
                #         l2.append(node2)
                #         l1.append(node2)  
                #         l2.append(node1)
                #         edges[(min(node1, node2), max(node1, node2))] = 1
                # data_mol.new_edge = torch.cat((torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0)), 0)
                data_mol.new_edge  = data_mol.new_edge_index
                #data_mol.edge_index = data_mol.new_edge 
                
                f = open('/home/wuxiang/GEC-main/time_count_rank_result_APPNP.txt', 'a')
                for method_no in range(4, 5):
                    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
                    #device = torch.device( 'cpu')
                    if dataname == "Cora":
                        args.num_features = 1433
                        args.num_classes = 7
                    elif dataname == "Citeseer":
                        args.num_features = 3703
                        args.num_classes = 6
                    elif dataname == "pubmed":
                        args.num_features = 500
                        args.num_classes = 3
                    elif dataname == "dblp":
                        args.num_features = 1639
                        args.num_classes = 4
                    elif dataname == "ogbn-arxiv":
                        args.num_features = data_mol.x.size()[1]
                        args.num_classes = 40
                    elif dataname == "Physics":
                        args.num_features = 8415
                        args.num_classes = 5
                    elif dataname == "ogbn-products":
                        args.num_features = data_mol.x.size()[1]
                        args.num_classes = 47
                   
                    # args.num_features = data_mol.x.size()[1]
                    # #args.num_classes
                    model = APPNP_Net(args).to(device)
                    # model = GCN(
                    #     in_channels=data.x.size(1),
                    #     hidden_channels=128,
                    #     out_channels=data.y.max().item() + 1,
                    #     num_layers=3,
                    #     dropout=0.5
                    # ).to(device)
                                        
                    if method_no == 4:
                        args.coarsening_method = 'Ours'
                    all_acc = []
                    all_val_loss = []
                    for i in range(args.runs):
                        train_label = torch.zeros(data_mol.new_x.shape[0], dtype = torch.int64)
                        train_mask = torch.zeros(data_mol.new_x.shape[0]).bool()
                        val_label = torch.zeros(data_mol.new_x.shape[0], dtype = torch.int64)
                        val_mask = torch.zeros(data_mol.new_x.shape[0]).bool()
                        val_m = {}
                        train_m = {}
                        # train_weight = torch.zeros([data_mol.new_x.shape[0], 7], dtype = torch.float).to(device)
                        # print(train_weight.shape)
                        for key in m.keys():
                            if data_mol.train_mask[key] == 1:
                                train_mask[m[key]] = True
                                if m[key] not in train_m.keys():
                                    train_m[m[key]] = []
                                train_m[m[key]].append(data_mol.y[key])
                            if data_mol.val_mask[key] == 1:
                                val_mask[m[key]] = True
                                if m[key] not in val_m.keys():
                                    val_m[m[key]] = []
                                val_m[m[key]].append(data_mol.y[key])
                        for key in train_m.keys():
                            #print(train_m[key])
                            # sub = sorted(train_m[key], key=lambda item: (train_m[key].count(item), item))
                            # if len(sub) > 1:
                            #    train_mask[key] = 0
                            # train_label[key] = sorted(train_m[key], key=lambda item: (train_m[key].count(item), item))[-1]
                            train_label[key] = data_mol.node_label[key]
                            # print(train_label[key])
                        for key in val_m.keys():
                            # sub = sorted(val_m[key], key=lambda item: (val_m[key].count(item), item))
                            # if len(sub) > 1:
                            #     val_mask[key] = 0
                            # val_label[key] = sorted(val_m[key], key=lambda item: (val_m[key].count(item), item))[-1]
                            val_label[key] = data_mol.node_label[key]
                        data = data_mol.to(device)
                        coarsen_features = data_mol.new_x.to(device)
                        # print(coarsen_train_labels.dtype)
                        # print(train_label.dtype)
                        coarsen_train_labels = train_label.to(device)
                        coarsen_train_mask = train_mask.to(device)
                        coarsen_val_labels = val_label.to(device)
                        coarsen_val_mask = val_mask.to(device)
                        coarsen_edge = data_mol.new_edge.to(device)
                        coarsen_edge = add_self_loops(coarsen_edge, num_nodes=coarsen_features.size(0))[0]
                        print((torch.unique(coarsen_train_labels[coarsen_train_mask],return_counts=True)))
                        print((torch.unique(coarsen_val_labels[coarsen_val_mask],return_counts=True)))

                        # homo = 0
                        # count =0
                        # for u in range(coarsen_edge.shape[1]):
                        #     if coarsen_train_mask[coarsen_edge[0][u]] and coarsen_train_mask[coarsen_edge[1][u]]:
                        #         count += 1

                        #         if coarsen_train_labels[coarsen_edge[0][u]] == coarsen_train_labels[coarsen_edge[1][u]]:
                        #             homo += 1
                        # print(f"train homo edge: {homo/count}")

                        # homo = 0
                        # count =0
                        # for u in range(coarsen_edge.shape[1]):
                        #     if coarsen_val_mask[coarsen_edge[0][u]] and coarsen_val_mask[coarsen_edge[1][u]]:
                        #         count += 1
                        #         if coarsen_val_labels[coarsen_edge[0][u]] == coarsen_val_labels[coarsen_edge[1][u]]:
                        #             homo += 1
                        # print(f"val homo edge: {homo/count}")

                        # if args.normalize_features:
                        #     coarsen_features = F.normalize(coarsen_features, p=1)
                        #     data.x = F.normalize(data.x, p=1)

                        model.reset_parameters()
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        criterion = torch.nn.CrossEntropyLoss()
                        best_val_loss = float('inf')
                        val_loss_history = []
                        for epoch in range(args.epochs):

                            model.train()
                            optimizer.zero_grad()
                            out = model(coarsen_features, coarsen_edge)
                            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                            #loss = criterion(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                            
                            # print(out[coarsen_train_mask].shape)
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            pred = model(coarsen_features, coarsen_edge)
                            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
                            #val_loss = criterion(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
                            # for n in coarsen_train_labels:
                            #     if n==-1:
                            #         print("err")
                            #print(val_loss)
                            if val_loss < best_val_loss and epoch > args.epochs // 2:
                                best_val_loss = val_loss
                                torch.save(model.state_dict(), path + 'checkpoint-best-acc2.pkl')
                                

                            val_loss_history.append(val_loss)
                            if args.early_stopping > 0 and epoch > args.epochs // 2:
                                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                                if val_loss > tmp.mean().item():
                                    break
                        all_val_loss.append(best_val_loss)
                        model.load_state_dict(torch.load(path + 'checkpoint-best-acc2.pkl'))
                        model.cpu()
                        model.eval()
                        data.x = data.x.cpu()
                        data.test_mask = data.test_mask.cpu()
                        data.y = data.y.cpu()
                        data.edge_index = data.edge_index.cpu()
                        pred = model(data.x, data.edge_index).max(1)[1]
                        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
                        print(test_acc)
                        all_acc.append(test_acc)
                    if len(all_acc) == 0:
                        f.write('%s  ' % args.coarsening_method)
                        f.write('unable to Coarse.' + '\n')
                        continue
                    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
                    print('val_loss: {:.4f}'.format(np.mean(all_val_loss)), '+/- {:.4f}'.format(np.std(all_val_loss)))
                    # f.write('%s  ' % args.coarsening_method)
                    f.write(f"dataset {dataname} , ratio {v}, num_edge {num_edge} deg1 {deg1} deg2 {deg2} \n ")
                    f.write('ave_acc: {:.4f}'.format(np.mean(all_acc)) + ' +/- {:.4f}'.format(np.std(all_acc)) + '\n')
                f.write('\n')
                f.close()

