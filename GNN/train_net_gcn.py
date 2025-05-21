import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net,GCN,net_gcn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull,Coauthor
from PyGdataset import PygNodePropPredDataset,Evaluator
import numpy as np
from torch_geometric.utils import to_undirected,add_self_loops
import os
#ArXiv dataset
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
embedding_dim_dict = {'Cora': [1433, 512, 7], 'Citeseer': [3703, 512, 6], 'pubmed': [500, 512, 3], 'ogbn-arxiv': [128, 512, 40],'dblp': [128, 512, 4]}
lr_dict = {'Cora': 0.01, 'Citeseer': 0.01, 'pubmed': 0.01, 'ogbn-arxiv': 0.01}
weight_decay_dict = {'Cora': 8e-5, 'Citeseer': 5e-4, 'pubmed': 5e-4, 'ogbn-arxiv': 5e-4}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    for dataname in ['ogbn-arxiv']:
        for num_edge in [60000]:
            # for v in [0.3, 0.5, 0.7, 0.8, 0.9]:
            for v in [0.5]:
            # for v in [0.8, 0.9]:
            # for v in [ 0.9]:
                #data_mol, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_offline.npy' % (dataname, v, rank), allow_pickle=True)
                #data_mol,m  = np.load('/home/wuxiang/GEC-main/Reduced_Node_Data/%s_%.2f_split%d_All_Simplex_1.npy' % (dataname, v, 1000), allow_pickle=True)
                data_mol, m = np.load('/home/wuxiang/strong_collapse/hyperparameter_study/%s_%.2f_%d_%d_%dsplit_All_Simplex_2.npy' % (dataname, v,num_edge,25,25), allow_pickle=True)
                #data_mol,m = np.load('/home/wuxiang/strong_collapse/Reduced_Node_Data/%s_%.2f_split_All_Simplex_1.npy' % (dataname, v), allow_pickle=True)

                #m = np.load('/home/wuxiang/strong_collapse/Reduced_Node_Data/%s_%.2f_split_All_Simplex_1.npy' % (dataname, v), allow_pickle=True)
                # data_mol, m = np.load('/home/ycmeng/ep2/Reduced_Node_Data/%s_%.2f_split%d_link_7.npy' % (dataname, v, rank), allow_pickle=True)
                #print(data_mol)
                if dataname in ['Cora','Citeseer','pubmed']:
                    dataset = Planetoid(root='/home/wuxiang/GEC-main/', name=dataname)
                    data = dataset[0]
                elif dataname in ['dblp','Physics']:
                    dataset = CitationFull(root='/home/wuxiang/GEC-main/dataset', name=dataname)
                    data = dataset[0]
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
                
                else:
                    dataset = Coauthor(root='/home/wuxiang/GEC-main/dataset', name=dataname)
                
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
                
                f = open('/home/wuxiang/GEC-main/time_count_rank_result.txt', 'a')
                #f.write('%d  %.2f\n' %  (rank, v))
                for method_no in range(4, 5):
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                   
                    # args.num_features = data_mol.x.size()[1]
                    # #args.num_classes
                    if args.model == 'net_gcn':
                        model_2 = net_gcn(embedding_dim=embedding_dim_dict[dataname]).to(device)
                    else:
                        model_2 = GCN(in_channels=args.num_features,hidden_channels=512,out_channels=args.num_classes,num_layers=3,dropout=0.5).to(device)
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
                        print(coarsen_features.shape)
                        # print(coarsen_train_labels.dtype)
                        # print(train_label.dtype)
                        coarsen_train_labels = train_label.to(device)
                        print(coarsen_train_labels.shape)
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

                        optimizer = torch.optim.Adam(model_2.parameters(), lr=lr_dict[dataname], weight_decay=weight_decay_dict[dataname])
                        best_val_acc = 0.0
                        best_test_acc = 0.0
                        loss_func = torch.nn.CrossEntropyLoss()
                        best_val_loss = float('inf')
                        val_loss_history = []
                        for epoch in range(args.epochs):

                            model_2.train()
                            optimizer.zero_grad()
                            output = model_2(coarsen_features, coarsen_edge,val_test =False )
                            loss = loss_func(output[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                            #loss = criterion(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                            
                            # print(out[coarsen_train_mask].shape)
                            loss.backward()
                            optimizer.step()

                            model_2.eval()
                            pred = model_2(coarsen_features, coarsen_edge,val_test=True)
                            y_pred = torch.log_softmax(pred, dim=-1)
                            y_pred = y_pred.argmax(dim=-1, keepdim=True)
                            valid_acc = evaluator.eval({
                                'y_true': coarsen_val_labels[coarsen_val_mask].unsqueeze(1),
                                'y_pred': y_pred[coarsen_val_mask],
                            })['acc']
                            #val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
                            #val_loss = criterion(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
                            # for n in coarsen_train_labels:
                            #     if n==-1:
                            #         print("err")
                            #print(val_loss)
                            #print(valid_acc)
                            if best_val_acc < valid_acc:
                                best_val_acc = valid_acc
                                torch.save(model_2.state_dict(), path + 'checkpoint-best-acc.pkl')
                                

                            val_loss_history.append(valid_acc)
                            if args.early_stopping > 0 and epoch > args.epochs // 2:
                                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                                if valid_acc < tmp.mean().item():
                                    break
                        #device1 = torch.device("cpu")
                        all_val_loss.append(best_val_loss)
                        model_2.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
                        #model_2.cpu()
                        model_2.eval()
                        data.x = data.x.to(device)
                        data.edge_index = data.edge_index.to(device)
                        data.y = data.y.to(device)
                        data.test_mask = data.test_mask.to(device)
                        pred = model_2(data.x, data.edge_index,val_test=True).max(1)[1]
                        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
                        print(test_acc)
                        all_acc.append(test_acc)
                    if len(all_acc) == 0:
                        f.write('%s  ' % args.coarsening_method)
                        f.write('unable to Coarse.' + '\n')
                        continue
                    f.write(f"dataset {dataname} , ratio {v}, num_edge {num_edge} \n")
                    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
                    print('val_loss: {:.4f}'.format(np.mean(all_val_loss)), '+/- {:.4f}'.format(np.std(all_val_loss)))
                    # f.write('%s  ' % args.coarsening_method)
                    f.write('ave_acc: {:.4f}'.format(np.mean(all_acc)) + ' +/- {:.4f}'.format(np.std(all_acc)) + '\n')
                f.write('\n')
                f.close()

