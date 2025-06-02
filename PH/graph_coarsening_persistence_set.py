import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from collections import deque
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from scipy.sparse import eye, coo_matrix,triu,diags

import time
import copy

from torch_geometric.utils import to_networkx
import random
import heapq
import tracemalloc
from scipy.sparse import csr_matrix
def draw_graph(edges,i): 
    # filtered_adj_matrix = collapsed_M[~deleted_node, :]  # 删除行
    # filtered_adj_matrix = filtered_adj_matrix[:, ~deleted_node]
    # collapsed_M = filtered_adj_matrix
    #print(num_collpsed_edges/2)

    vis_M = nx.Graph()
    vis_M.add_edges_from(edges)
    dict(vis_M.degree())
    plt.figure(figsize=(30, 30))
    nx.draw(vis_M, with_labels=True, node_color='lightblue', node_size=100, font_size=1)
    plt.savefig("collapsed_graph_figure_"+str(i)+".png", format="png")

#edges = torch.tensor([[0, 1, 2, 3],[0,2,3,1]])
class nodes:
    def __init__(self, index):  # 初始化 包含左端点所在的x, y坐标以及长度l
        self.index = index
        self.vanished = 0  # 这个点上重叠的点数
        self.edgenode = 0  # 这个点上的边数
        self.nodes = []  # 这个点上重叠的点的列表
        self.ed_van = []  # 这个点上由于消去边获得的特征
        self.train_node = False  # 是否为训练节点
        self.remain = False  # 是否需要保留
        self.recast = 0  # 是否为训练节点
        self.label = -1
       

    def __lt__(self, other):  # 为了堆优化重载运算符"<"
        if self.edgenode < other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast < other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished < other.vanished:
                return True
            return False

    def __gt__(self, other):  # 为了堆优化重载运算符">"
        if self.edgenode > other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast > other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished > other.vanished:
                return True
            return False

    def __eq__(self, other):  # 为了堆优化重载运算符"="
        if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished == other.vanished:
            return True
        else:
            return False

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def compute_arboricity(G):
    # Step 1: Initialize arboricity count and edge set
    arboricity = 0
    edges = set(G.edges())
    
    while edges:
        # Step 2: Find the maximum spanning tree (using any algorithm like DFS or BFS)
        spanning_tree = nx.maximum_spanning_tree(G.subgraph([u for u, v in edges]))
        
        # Step 3: Remove the edges of the current spanning tree from the edge set
        edges -= set(spanning_tree.edges())
        
        # Increment the arboricity count
        arboricity += 1
    
    return arboricity

def find_component(data):
    n = []
    graph = nx.Graph()
    for i in range(len(data.x)):
        graph.add_node(i)
        n.append(nodes(i))
        if keep_mask[i] == 1:
            n[i].train_node = True
            n[i].remain = True
        # if data.train_mask[i] == 1:
        #     n[i].label = int(data.y[i])
            # n[i].remain = True
        # if data.val_mask[i] == 1:
        # if random.random() > 0.5:
        # n[i].train_node = True
        # n[i].remain = True
        #     print('train_node', i)
    # n[2].train_node = False
    # n[3].train_node = False
    # n[0].train_node = False
    # n[5].train_node = False
    # print(graph.nodes())
    for i in range(len(data.edge_index[0])):
        n1n1 = int(data.edge_index[0, i])
        n2n2 = int(data.edge_index[1, i])
        if n1n1 != n2n2:
            graph.add_edge(n1n1, n2n2)
        # if int(data.edge_index[0, i]) == 0 or int(data.edge_index[1, i]) == 0:
        #     print("???")
        # n[int(data.edge_index[0, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
        # n[int(data.edge_index[1, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
    # print(graph.nodes())
    # print(graph.edges())
    cnt_cluster_node = 0
    num_nodes = data.x.size()[0]
    n_vanished = 0
    
    #print(compute_arboricity(graph))
    print("time:", time.time_ns())
    com_nodes = []
    components = nx.connected_components(graph)
    print(components)
    for component in components:
        com = list(component)
        # print("components", len(com), com)
        if len(com) >= 10:
            com_nodes = com_nodes + com
        else:
            add_flag = 0
            for node in com:
                if n[node].remain:
                    add_flag = 1
                    break
            if add_flag == 1:
                com_nodes = com_nodes + com
            else:
                for node in com:
                    n[node].vanished = True
                n_vanished += len(com)
    #graph = nx.Graph(graph.subgraph(com_nodes))
    #old_graph = nx.Graph(graph)
    return com_nodes

def build_sparse_adjacency_matrix(edge_list, num_nodes):
    rows = edge_list[0]
    cols = edge_list[1]

    
    
    
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
    
   

# 1. 加载 ogbn-arxiv 数据集








# edges_reverse = edges[[1,0],:]
# edges = np.concatenate([edges, edges_reverse], axis=1)

# edges = edges.T
# # edges =  np.unique(edges, axis=0)
# adj_list = [[i] for i in range(num_nodes)]
# print(edges.shape[0])
# for i in range(edges.shape[0]):
#     adj_list[edges[i][0]].append(edges[i][1])
#     adj_list[edges[i][1]].append(edges[i][0])









#G = to_networkx(data, to_undirected=True)


# # G = nx.Graph()
# # G.add_nodes_from([0,1, 2, 3, 4])
# # G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
# num_edges = G.number_of_edges()
# num_nodes  = G.number_of_nodes()
# M= np.zeros((num_edges, num_nodes), dtype=int)

#G = nx.erdos_renyi_graph(50, 0.05)
#G = nx.cycle_graph(8)




# plt.figure(figsize=(10, 10))
# nx.draw(G, with_labels=True, node_color='lightblue', node_size=100, font_size=16)
# plt.savefig("graph_figure.png", format="png")
# 如果想转换为 NumPy 数组

# b1,b2 = compute_betti_numbers(adj)
# print(f"Betti 1 (环的数量): {b1}")
# print(f"Betti 2 (空腔的数量): {b2}")

# a test graph comment if you use public dataset
# node_num = 30
# num_nodes = node_num
# G = nx.barabasi_albert_graph(node_num, 2)
# edges = np.array(list(G.edges())).T
# #print(edges)

# edges = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 10, 10, 13, 15, 16, 17, 18, 19, 21, 22, 23],
#     [1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 3, 5, 7, 10, 25, 21, 4, 6, 15, 17, 22, 24, 9, 15, 17, 23, 26, 12, 19, 8, 18, 8, 12, 13, 14, 16, 21, 23, 27, 28, 29, 10, 11, 25, 29, 11, 18, 14, 19, 24, 22, 20, 20, 27, 26, 28]]
# )
# G = nx.Graph()
# G.add_edges_from(edges.T)
# adj = nx.adjacency_matrix(G)
# adj = adj + eye(node_num, format='coo')
# keep_nodes = [i for i in range(node_num)]
# print(f"original graph {node_num} nodes, {G.number_of_edges()} edges.")

# plt.figure(figsize=(10, 10))
# nx.draw(G, with_labels=True, node_color='lightblue', node_size=100, font_size=16)
# plt.savefig("graph_figure.png", format="png")





    





def visual_graph(M,save_name):
    vis_M = M.copy() - np.eye(M.shape[0])
    non_zero_rows = ~np.all(vis_M == 0, axis=1)
    non_zero_cols = ~np.all(vis_M == 0, axis=0)
    vis_M_collapsed = vis_M[non_zero_rows][:, non_zero_cols]
    vis_M_collapsed = nx.from_numpy_array(vis_M_collapsed)
    plt.figure(figsize=(10, 10))
    nx.draw(vis_M_collapsed, with_labels=True, node_color='lightblue', node_size=100, font_size=16)
    plt.savefig(save_name, format="png")










class CoreAlgorithm:
    def __init__(self, M, node_degree, filtration_value_dict,keep_nodes,reduction_ratio,dataname = "default",save = True,degree_threshold = 5,degree_threshold2 = 5,require_edge = True):      

        #tracemalloc.start(
        self.M = M
        #self.original_M = M.copy()
        # self.adj_2_hop = M @ M
        #self.adj_2_hop = self.adj_2_hop.tolil()
        
        self.keep_nodes = keep_nodes
       
        self.list_of_set = [set(row) for row in M.rows]
        self.require_edge = require_edge
        self.delete_obj = len(self.list_of_set)*reduction_ratio
        

        self.deleted_to_remain = dict(zip(keep_nodes, keep_nodes))
        self.num_remain_nodes = len(self.keep_nodes)
        
        self.deleted_node =  np.full(M.shape[0], False, dtype=bool)
  
  
        #draw_graph(adj,-1)
        self.dataname = dataname
        # for i in keep_nodes:
        #     self.deleted_node[i] = False

        self.filtration_value_dict = filtration_value_dict        

        #M_csr = M.tocsr()
        self.node_degree = node_degree
        #self.node_degree = np.array([M.getrow(i).nnz for i in range(M.shape[0])])
        #self.need_delete = deque()
        #self.node_queue = deque(keep_nodes)
        self.edge_queue = deque()
        self.round = 0
        self.finish = False
        self.insert_dominated_edge = False
        #self.keep_mask = ~(copy.deepcopy(self.deleted_node))

        self.heruistic_delete = False
        #self.label = node_label
        self.heruistic_delete_deg3 = False
        self.save = save
       

        #self.old_to_new = dict()
        #self.new_to_old = dict()
        
       
        self.strong_relaxed = False
        
        self.finish1 = False
        self.finish2 = False

        self.remain_edges = set()

        #self.remain_edges = self.get_all_edges_set()
        edges = self.get_all_edges()
        for e in edges:    
            self.remain_edges.add(e)
        
        self.edge_num = len(self.remain_edges)*2
       
        self.isfirst = True
        self.finish_reduction = False
        self.strong_tolerance = 0
        self.degree_threshold = degree_threshold
        self.degree_threshold2 = degree_threshold2
        self.insert_dominated_node= dict()
        #self.filters = {}
        #self.initilize_bloom_filter()
        #current, peak = tracemalloc.get_traced_memory()
    
        peak = 0
        #tracemalloc.stop() 
        self.max_mem = peak
        #self.inserted_edges = {}
    
    def get_filtration_value_dict(self):
        return self.filtration_value_dict
    # def initilize_bloom_filter(self):
    #     edges = self.get_all_edges()
    #     for e in edges:
    #         for node in [e[0], e[1]]:
    #             if self.node_degree[node] > 10:
    #                 if node not in self.filters:
    #                     self.filters[node] = CountingBloomFilter(1000, 1)
    #                 self.filters[node].add(str(e[0] if node == e[1] else e[1]))
        
    def swap_coarsened_graph(self,new_to_old,old_to_new,edges):  

        
     
        num_nodes = len(new_to_old)
        rows = edges[0]
        cols = edges[1]

        print(f"density : {len(edges[0])/num_nodes}")
    
        edges = edges.T
    
        new_adj_mat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
        new_adj_mat = new_adj_mat.tolil()
        supernode_label = dict()
        supernode_label_list = dict()

        for key,value in new_to_old.items():
            
            supernode_label_list[key] = self.node_label[value].numpy()
            

        for key,value in supernode_label_list.items():
            values, counts = np.unique(value[0], return_counts=True)

            # 找出出现次数最多的元素
            max_label = value[np.argmax(counts)]
            supernode_label[key] = max_label

        homo = 0
        for e in  edges:
            if supernode_label[e[0]] == supernode_label[e[1]]:
                    homo += 1
        print(f"homo edge: {homo/len(edges)}")
        
        for e in edges:
            u = e[0]
            v = e[1]
            if supernode_label[u] != supernode_label[v]:
                # compare swaped homophily
                homo_swap = 0
                homo_original = 0
                for i in new_adj_mat.rows[u]:
                    if supernode_label[i] == supernode_label[v]:
                        homo_swap += 1
                    if supernode_label[i] == supernode_label[u]:
                        homo_original += 1
                for i in new_adj_mat.rows[v]:
                    if supernode_label[i] == supernode_label[u]:
                        homo_swap += 1
                    if supernode_label[i] == supernode_label[v]:
                        homo_original += 1
                if homo_original <homo_swap:
                    t = supernode_label[u]
                    supernode_label[u] = supernode_label[v]
                    supernode_label[v] = t
                    t = new_to_old[u] 
                    new_to_old[u]  = new_to_old[v]
                    new_to_old[v] = t
        homo = 0
        for u in range(new_adj_mat.shape[0]):
            for v in new_adj_mat.rows[u]:
                if supernode_label[u] == supernode_label[v]:
                    homo += 1
        print(f"homo edge: {homo/len(edges)}")

        for key,value in new_to_old.items():
            for v in value:
                old_to_new[v] = key
        return new_to_old,old_to_new,supernode_label
    
    def set_filtration_value_dict(self,filtration_value_dict):
        self.filtration_value_dict = filtration_value_dict
    
    def approximate_persistence(self):
        #self.run_algorithm_preserve_homotopy()
        
        self.node_queue = deque()
        self.edge_queue = deque()
        persistent_diagram = []
        
        edges = self.get_all_edges()
        # 按照filtration_value_dict 的值对edges进行降序排序
        edges = sorted(edges, key=lambda edge: self.filtration_value_dict[edge], reverse=True)
        edge_queue = deque(edges)
        index = 0
        while edge_queue:
            e = edge_queue.popleft()
            if self.filtration_value_dict[e] == 8:
                print("5")
            if self.M[e[0],e[1]] ==0:
                continue

            self.M[e[0],e[1]] = False
            self.M[e[1],e[0]] = False
            self.node_degree[e[0]] -= 1
            self.node_degree[e[1]] -= 1

            persistent_diagram.append((1,(self.filtration_value_dict[e],np.inf)))

            self.node_queue.append(e[0])
            self.node_queue.append(e[1])

            self.strong_collapse()
            self.edge_collapse()
            index += 1
            #self.draw_graph("collapse_"+str(index))

        return persistent_diagram

    # def delete_heterophlic_edge(self,new_to_old,old_to_new,edges):
    #     num_nodes = len(new_to_old)
    #     rows = edges[0]
    #     cols = edges[1]
    #     delete_edges = dict()

    #     print(f"density : {len(edges[0])/num_nodes}")
    
    #     edges = edges.T

    

        
    #     supernode_label = dict()
    #     supernode_label_list = dict()
    #     real_supernode_label_list = dict()
    #     real_supernode_label = dict()

    #     supernode_train_mask = [False for i in range(len(new_to_old))]
 
    #     for key,value in new_to_old.items():
    #         for v in value:
    #             if not self.test_mask[v]:
    #                 supernode_train_mask[key] = True
    #                 break
            
    #         value_filtered = [v for v in value if not self.test_mask[v]]
    #         if len(value_filtered) / len(value) > 0.5:
    #             supernode_label_list[key] = self.label[value_filtered]
    #         else:
    #             supernode_label_list[key] = []
    #         real_supernode_label_list[key] = self.label[value]
    #     # for key,value in new_to_old.items():
            
    #     #     supernode_label_list[key] = self.label[value].numpy()
            
    #     # for key,value in supernode_label_list.items():
    #     #     values, counts = np.unique(value, return_counts=True)

    #     #     # 找出出现次数最多的元素
    #     #     max_label = value[np.argmax(counts)]
    #     #     supernode_label[key] = max_label

    #     for key,value in real_supernode_label_list.items():
            
    #         values, counts = np.unique(value, return_counts=True)

    #         # 找出出现次数最多的元素
    #         max_label = value[np.argmax(counts)]
    #         real_supernode_label[key] = max_label


    #     for key,value in supernode_label_list.items():
    #         if len(value) == 0:
    #             supernode_label[key] = -1
    #             continue
    #         values, counts = np.unique(value, return_counts=True)

    #         # 找出出现次数最多的元素
    #         max_label = value[np.argmax(counts)]
    #         supernode_label[key] = max_label

    #     for key,value in supernode_label.items():
    #         if value == -1:
                
    #             # values, counts = np.unique(label_list, return_counts=True)
    #             # max_label = values[np.argmax(counts)]
    #             # supernode_label[key] = max_label
    #             # if supernode_label[key] == -1:
    #             supernode_label[key] =0
    #     delete_edges = set()

    #     homo = 0
    #     for e in  edges:
    #         if real_supernode_label[e[0]] == real_supernode_label[e[1]]:
    #                 homo += 1
    #     print(f"homo edge: {homo/len(edges)}")

        
    #     new_edges = []
    #     delete = 0
    #     for e in edges:

    #         if supernode_train_mask[e[0]]  and supernode_train_mask[e[1]] and supernode_label[e[0]] != supernode_label[e[1]] :
               
    #             delete += 1
    #         else:
    #             new_edges.append(e)
    #     print(f"delete {delete} edges")
          
                
        
    #     homo = 0
    #     count = 0
        
    #     for e in  new_edges:
            
    #         if real_supernode_label[e[0]] == real_supernode_label[e[1]] :
    #                 homo += 1
        
    #     self.supernode_label = supernode_label
    #     print(f"homo edge: {homo/len(new_edges)}")
    #     new_edges = np.array(new_edges)
    #     return new_edges,supernode_label




    def delete_heterophlic_edge(self,obj):
        edges= self.get_all_edges()
        #random.shuffle(edges)
        obj_num = obj
    

    

        node_label = dict()
 
        

        for key,value in enumerate(self.label_list):
            
            values, counts = np.unique(value, return_counts=True)
            if len(values) == 0:
                node_label[key] = -1
            else:
            # 找出出现次数最多的元素
                max_label = values[np.argmax(counts)]
                node_label[key] = int(max_label)

        
        self.supernode_label1 = node_label

        
        

        
        count = 0
        homo = 0
        edges= self.get_all_edges()
        for e in  edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1
        
        print(f"homo edge: {homo/count}")

        
        delete = 0
        
        for e in edges:
            
            if  node_label[e[0]] != node_label[e[1]] and  node_label[e[0]] != -1 and node_label[e[1]] != -1: 
                
                self.M[e[0],e[1]] = False
                self.M[e[1],e[0]] = False
                delete +=1 
                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1 
                for neighbor in self.M.rows[e[0]]:
                    self.potential_node_set.add(neighbor)
                for neighbor in self.M.rows[e[1]]:
                    self.potential_node_set.add(neighbor)
                if delete == obj_num:
                    break
        
        # contain_num = torch.tensor(self.conain_node)
        # contain_num =contain_num.unsqueeze(1)
        # average_feature =self.node_feature / contain_num

        # degrees = self.M.sum(axis=1).A1  # .A1 将结果转换为 1D 数组
        # degrees = 1.0/ degrees
        # # 构造度矩阵（对角稀疏矩阵）
        # degree_matrix = diags(degrees)
        
        # average_feature = self.M @ average_feature 
        # # for i in self.node_feature.shape[0]:
        # #     self.node_feature[i] 
        # average_feature =   degree_matrix @   average_feature
        # average_feature = torch.from_numpy(average_feature)
        # if delete < obj_num:
        #     edges= self.get_all_edges()
        #     for e in  edges:
        #         if torch.cosine_similarity(average_feature[e[0]],average_feature[e[1]],dim=0) < 0.2:
        #             self.M[e[0],e[1]] = False
        #             self.M[e[1],e[0]] = False
        #             self.node_degree[e[0]] -= 1
        #             self.node_degree[e[1]] -= 1 
        #             delete +=1 
        #             for neighbor in self.M.rows[e[0]]:
        #                 self.potential_node_set.add(neighbor)
        #             for neighbor in self.M.rows[e[1]]:
        #                 self.potential_node_set.add(neighbor)
        #             if delete > obj_num:
        #                 break

        # print(f"delete {delete} edges")
        print(f"delete {delete} edges")  
                
        
        homo = 0
        count = 0
        edges= self.get_all_edges()
        for e in  edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1
        
        print(f"homo edge: {homo/count}")
        return  
        
    
    def swap_heterophlic_edge(self,new_to_old,old_to_new,edges):  

        
     
        num_nodes = len(new_to_old)
        rows = edges[0]
        cols = edges[1]

        print(f"density : {len(edges[0])/num_nodes}")
    
        edges = edges.T
    
        new_adj_mat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
        new_adj_mat = new_adj_mat.tolil()
        supernode_label = dict()
        supernode_label_list = dict()

        for key,value in new_to_old.items():
            
            supernode_label_list[key] = self.label[value].numpy()
            

        for key,value in supernode_label_list.items():
            values, counts = np.unique(value[0], return_counts=True)

            # 找出出现次数最多的元素
            max_label = value[np.argmax(counts)]
            supernode_label[key] = max_label

        homo = 0
        for e in  edges:
            if e[0] > 900:
                print(900)
            if supernode_label[e[0]] == supernode_label[e[1]]:
                    homo += 1
        print(f"homo edge: {homo/len(edges)}")

        for e in edges:
            u = e[0]
            v = e[1]
            if supernode_label[u] != supernode_label[v]:
                swap =False
                for n in new_adj_mat.rows[u]:
                    if supernode_label[n] != supernode_label[u] and supernode_label[n] == supernode_label[v]:
                        t = supernode_label[u]
                        supernode_label[u] = supernode_label[n]
                        supernode_label[n] = t
                        t = new_to_old[u] 
                        new_to_old[u]  = new_to_old[n]
                        new_to_old[n] = t
                        swap =True
                        break
                if swap == False:
                    for n in new_adj_mat.rows[v]:
                        if supernode_label[n] != supernode_label[v] and supernode_label[n] == supernode_label[u]:
                            t = supernode_label[v]
                            supernode_label[v] = supernode_label[n]
                            supernode_label[n] = t
                            t = new_to_old[v] 
                            new_to_old[v]  = new_to_old[n]
                            new_to_old[n] = t
                            swap =True
                            break
                
                        
                    
        
        
        
                   
        homo = 0
        for u in range(new_adj_mat.shape[0]):
            for v in new_adj_mat.rows[u]:
                if supernode_label[u] == supernode_label[v]:
                    homo += 1
        print(f"homo edge: {homo/len(edges)}")

        for key,value in new_to_old.items():
            for v in value:
                old_to_new[v] = key
        return new_to_old,old_to_new,supernode_label

    def is_dominated_edge_return_node(self,e):
        NG_e0 = self.list_of_set[e[0]]
            
        NG_e1 = self.list_of_set[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,e)
        #NG_e = [n for n in NG_e if (n not in e and self.deleted_node[n] == False)]
        NG_e = [n for n in NG_e if n not in e ]
        if len(NG_e) == 0:
            return False,0,NG_e
        if len(NG_e) == 1:
            return True,NG_e[0],NG_e
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False,0,NG_e
        for w in NG_e:
            is_dominated=True
            #NG_w = self.M.rows[w]
            for n in NG_e:
                if w not in self.list_of_set[n]:
                    is_dominated = False
                    break
                
            if is_dominated == False:
                continue
            else:
                is_dominated = True
                return True,w,NG_e
        if is_dominated == False:
            return False,0,NG_e
    
    
    def initilized_node_label_dict(self,new_to_old,old_to_new,edges):
        num_nodes = len(new_to_old)
        rows = edges[0]
        cols = edges[1]

        print(f"edge num : {len(edges[0])}")
    
    
    
        new_adj_mat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
        new_adj_mat = new_adj_mat.tolil()
        supernode_label = dict()
        supernode_label_list = dict()

        
        for key,value in new_to_old.items():
            
            supernode_label_list[key] = self.node_label[value].numpy()
            

        for key,value in supernode_label_list.items():
            values, counts = np.unique(value[0], return_counts=True)

            # 找出出现次数最多的元素
            max_label = value[np.argmax(counts)]
            supernode_label[key] = max_label
        
        homo = 0
        for u in range(new_adj_mat.shape[0]):
            for v in new_adj_mat.rows[u]:
                if supernode_label[u] == supernode_label[v]:
                    homo += 1
        print(f"homo edge: {homo/2}")


        count  = 0
        for u in range(num_nodes):
            
            anomoly_node= []
            for n in new_to_old[u]:
                if self.node_label[n] != supernode_label[u]:
                    anomoly_node.append(n)
            for n in anomoly_node:
                for k in new_adj_mat.rows[u]:
                    if supernode_label[k] == self.node_label[n]:
                        count += 1
                        new_to_old[k].append(n)
                        new_to_old[u].remove(n)
                        old_to_new[n] = k
                        break
        homo = 0
        for u in range(new_adj_mat.shape[0]):
            for v in new_adj_mat.rows[u]:
                if supernode_label[u] == supernode_label[v]:
                    homo += 1
        print(f"homo edge: {homo/2}")
                
        print(f"reallocated node {count}")
        return new_to_old,old_to_new

    def draw_graph(self,save_name): 
        edges = self.get_all_edges()
        print(len(edges))
        
        vis_M = nx.Graph()
        vis_M.add_edges_from(edges)
        dict(vis_M.degree())
        # subgraph_nodes = [174]
        # seed_nodes = 174
        # for n in vis_M.neighbors(seed_nodes):
        #     subgraph_nodes.append(n)
        #     for j in vis_M.neighbors(n):
        #         subgraph_nodes.append(j)
        #         for i in vis_M.neighbors(j):
        #             subgraph_nodes.append(i)

        #subgraph = vis_M.subgraph(subgraph_nodes)
        subgraph.number_of_edges()
        plt.figure(figsize=(10, 10))
        nx.draw(subgraph, with_labels=True, node_color='lightblue', node_size=20, font_size=10)
        plt.savefig(save_name)


    def sorted_list_intersection_with_deleted_set(self, list1, list2,e):
        
        intersection = np.empty(len(list1),dtype=int)
        count = 0 
        n, m = len(list1), len(list2)
        # 使用双指针遍历两个列表
        short_list = list1 if n<m else list2
        w = e[1] if n<m else e[0]
        for node in short_list:
            if self.deleted_node[node]:
                continue
            if node in self.list_of_set[w]:
                intersection[count] = node
                count += 1
        return intersection[:count]
    def  sorted_list_intersection_with_deleted(self, list1, list2):
        i, j = 0, 0
        intersection = np.empty(len(list1),dtype=int)
        count = 0 
        n, m = len(list1), len(list2)
        # 使用双指针遍历两个列表
        while i < n and j < m:
            # 跳过 list1 中被标记删除的元素
            while i < n and self.deleted_node[list1[i]]:
                i += 1
            
            # 跳过 list2 中被标记删除的元素
            while j < m and self.deleted_node[list2[j]]:
                j += 1
            
            # 如果其中一个列表已经遍历完，结束循环
            if i >= n or j >= m:
                break
            
            if list1[i] == list2[j]:
                
                intersection[count] = list1[i]
                count += 1
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1  # 移动 list1 的指针
            else:
                j += 1  # 移动 list2 的指针
        
        return intersection[:count]
    

    def is_subset_with_deleted(self,sorted_list1, sorted_list2):
        
        i, j = 0, 0
        n, m = len(sorted_list1), len(sorted_list2)

        while i < n:
            # 找到 sorted_list1 中下一个有效元素
            while i < n and self.deleted_node[sorted_list1[i]]:
                i += 1
                
            # 找到 sorted_list2 中下一个有效元素
            while j < m and self.deleted_node[sorted_list2[j]]:
                j += 1
            
            if i >= n:  # list1 已遍历完
                return True
            
            if j >= m:  # list2 已遍历完，但 list1 还没结束
                return False

            if sorted_list1[i] < sorted_list2[j]:
                return False  # list1 的当前元素小于 list2 的当前元素
            elif sorted_list1[i] == sorted_list2[j]:
                i += 1  # 移动 list1 的指针
            j += 1  # 始终移动 list2 的指针

        return True  # list 
    


    def is_subset_with_deleted_set(self, sorted_list1, sorted_list2,w):
        #flag = True 
        count = 0
        for n_list1 in sorted_list1:
            if self.deleted_node[n_list1] == False and n_list1 not in self.list_of_set[w]:
                return False
        return True
    def is_subset_with_deleted_Tolerance(self, sorted_list1, sorted_list2,tolerance):
        i, j = 0, 0
        n, m = len(sorted_list1), len(sorted_list2)
        mismatch_count = 0  # 记录不匹配的元素数量

        while i < n:
            # 找到 sorted_list1 中下一个有效元素
            while i < n and self.deleted_node[sorted_list1[i]]:
                i += 1

            # 找到 sorted_list2 中下一个有效元素
            while j < m and self.deleted_node[sorted_list2[j]]:
                j += 1

            if i >= n:  # list1 已遍历完
                return True

            if j >= m:  # list2 已遍历完，但 list1 还没结束
                mismatch_count += 1
                if mismatch_count > tolerance:
                    return False
                i += 1
                continue

            if sorted_list1[i] < sorted_list2[j]:
                mismatch_count += 1
                if mismatch_count > tolerance:
                    return False
                i += 1
            elif sorted_list1[i] == sorted_list2[j]:
                i += 1
            j += 1  # 始终移动 list2 的指针

            # 如果不匹配的元素超过 1 个，返回 False
            

        return True

    
    
    # def strong_collapse(self):
    #     st = time.time()
    #     #node_queue = deque(keep_nodes)
        
    #     pushed_nodes = set(list(self.node_queue))
    #     #print(2034 in pushed_nodes)
        
    #     collapse_count = dict()
    #     st_time = time.time()
    #     self.deleted_node_list = dict()
       
    #     last_num_nodes = self.num_remain_nodes
    #     print(f"round {self.round} strong collapse, node_num {len(self.node_queue)}")
    #     while self.node_queue:
    #         self.round += 1
    #         if self.round % 10000 == 0:
    #             ed = time.time()
    #             print("round {} strong collapse: {}".format(round,ed-st))
    #             st = ed
    #             print(len(self.node_queue))

    #         v = self.node_queue.popleft()
            
    #         pushed_nodes.remove(v)
    #         if self.isfirst == False and self.heruistic_delete == False and v not  in self.potential_node_set:
    #             continue
    #         if  self.node_degree[v]>10:
    #             continue

            
            
            
           
    #         if self.deleted_node[v]:
    #             continue
            
    #         NG_v = self.M.rows[v]

            
            
            
    #         for w in NG_v:
    #             if w == v or self.deleted_node[w] or self.node_degree[w] < self.node_degree[v]:
    #                 continue
    #             if self.node_degree[v] > 3 and self.node_degree[v] <  self.node_degree[w]:
    #                 continue 
    #             NG_w = self.M.rows[w]

    #             #if len(self.sorted_list_intersection_with_deleted(NG_v, NG_w)) == self.node_degree[v]:
                
    #             if self.is_subset_with_deleted_Tolerance(NG_v, NG_w,self.strong_tolerance):
                    
                    
                    
    #                 if self.node_degree[v]>3:
    #                     self.deleted_node_list[v] =w
    #                 self.deleted_node[v] = True
    #                 self.need_delete.append(v)
    #                 self.node_degree[v] = 1
    #                 self.deleted_to_remain[v] = w
                    
               
    #                 self.num_remain_nodes -= 1
                    

    #                 for neighbor in NG_v:
    #                     if not self.deleted_node[neighbor]:
    #                         self.node_degree[neighbor] -= 1
                            
    #                         #self.potential_node.append(neighbor)
    #                         if neighbor not in pushed_nodes:
    #                             pushed_nodes.add(neighbor)
    #                             self.node_queue.append(neighbor)
    #                 break
                
           
    #     # save self.deleted_to_remain[v]
    #     #np.save("deleted_to_remain_1.npy",np.array(self.deleted_to_remain))
    #     #self.deleted_node_list = self.need_delete.copy()
    #     self.delayed_delete()
    #     #self.potential_node_set = set(self.potential_node)
    #     #self.potential_node = []
    #     num_remain_nodes = self.num_remain_nodes
    #     ed_time = time.time()
    #     print(f"round {self.round} strong collapse: {ed_time-st_time}s, reduce_nodes total {len((self.keep_nodes))-self.num_remain_nodes}, reduce nodes this round {last_num_nodes-num_remain_nodes} remain_nodes {self.num_remain_nodes}")
    #     if last_num_nodes - num_remain_nodes < 1:
    #         self.insert_dominated_edge = True
            
    #         if self.heruistic_delete:
    #             self.strong_tolerance+=1
    #             print(f"strong tolerance +1 {self.strong_tolerance}")
            
    #         return 
    #     else:
    #         self.finish	 = False
    #         return 
    
    def strong_collapse(self):
        st = time.time()
        #node_queue = deque(keep_nodes)
        
        pushed_nodes = set(list(self.node_queue))
        #print(2034 in pushed_nodes)
        
        collapse_count = dict()
        st_time = time.time()
       
        last_num_nodes = self.num_remain_nodes
        #print(f"round {self.round} strong collapse, node_num {len(self.node_queue)}")
        while self.node_queue:
            # self.round += 1
            # if self.round % 10000 == 0:
            #     ed = time.time()
            #     print("round {} strong collapse: {}".format(round,ed-st))
            #     st = ed
            #     print(len(self.node_queue))

            v = self.node_queue.popleft()
            if self.deleted_node[v]:
                continue
            pushed_nodes.remove(v)
           
            if  self.node_degree[v]>self.degree_threshold:
                continue

            if self.num_remain_nodes < self.delete_obj:
                break
            
           
            if self.deleted_node[v]:
                continue
            
            NG_v = self.list_of_set[v]

            
            
            
            for w in NG_v:
                if w == v or self.deleted_node[w] or self.node_degree[w] < self.node_degree[v]:
                    continue
                # if self.node_degree[v] > 3 :
                #     continue 
                NG_w = self.list_of_set[w]

                #if len(self.sorted_list_intersection_with_deleted(NG_v, NG_w)) == self.node_degree[v]:
                
                if self.is_subset_with_deleted_set(NG_v, NG_w,w) :
                    # if self.node_degree[v] > 3:
                    #     print(v)
                    
                    if self.node_degree[v]>3 and v < w :
                        #print(self.node_degree[v])
                        
                        
                        
                        for a in NG_v:
                            if a == w or a==v:
                                continue
                            if (w,a) in self.filtration_value_dict:
                                if self.filtration_value_dict[(w,a)] > v:
                                    self.filtration_value_dict[(w,a)] = v
                                    self.filtration_value_dict[(a,w)] = v
                        # for i in range(len(ng)):
                        #     for j in range(i+1,len(ng)):
                        #         if ng[j] in self.M.rows[ng[i]]:
                        #             if (ng[i],ng[j]) in self.filtration_value_dict:
                        #                 other_node = ng[i] if ng[j] == w else ng[j]
                        #                 if self.filtration_value_dict[(ng[i],ng[j])] > v:
                        #                     self.filtration_value_dict[(ng[i],ng[j])] = v
                        #                     self.filtration_value_dict[(ng[j],ng[i])] = v
                    self.deleted_node[v] = True
                    #self.need_delete.append(v)
                    self.node_degree[v] = 1
                    self.deleted_to_remain[v] = w
                    
               
                    self.num_remain_nodes -= 1
                    

                    for neighbor in NG_v:
                        if not self.deleted_node[neighbor]:
                            self.node_degree[neighbor] -= 1
                            # if self.isfirst == False:
                            #     self.potential_node.add(neighbor)
                            #self.potential_node.append(neighbor)
                            if neighbor not in pushed_nodes:
                                pushed_nodes.add(neighbor)
                                self.node_queue.append(neighbor)
                    break
                
           
        # save self.deleted_to_remain[v]
        #np.save("deleted_to_remain_1.npy",np.array(self.deleted_to_remain))
        #self.deleted_node_list = self.need_delete.copy()
        #self.delayed_delete()
        #self.potential_node_set = set(self.potential_node)
        #self.potential_node = []
        #self.potential_edge = set()
        num_remain_nodes = self.num_remain_nodes
        ed_time = time.time()
        print(f"round {self.round} strong collapse: {ed_time-st_time}s, reduce_nodes total {len((self.keep_nodes))-self.num_remain_nodes}, reduce nodes this round {last_num_nodes-num_remain_nodes} remain_nodes {self.num_remain_nodes}")
        # if last_num_nodes - num_remain_nodes < 10:
        #     self.insert_dominated_edge = True
            
        #     if self.heruistic_delete:
        #         self.strong_tolerance+=1
        #         print(f"strong tolerance +1 {self.strong_tolerance}")
            
        #     return 
        # else:
        #     self.finish	 = False
        #     return 
        
    
    


    def edge_collapse(self):
        t0 = time.time()
        edges = list(self.remain_edges)
        self.edge_queue = deque(edges)
        pushed_edges = set(self.edge_queue)
        round = 0
        intersection_time = 0
        st = time.time()
        count = dict()
        dominate_count = dict()
        deleted_edges = 0
        
        # edges = self.get_all_edges()
        # self.edge_queue = deque(edges)
        # print(len(self.edge_queue))
        #delelted_edges_list = []
        while self.edge_queue:
            round += 1
            if round % 1000000 == 0:
                ed = time.time()
                print("round {} edge collapse: {} intersection time {} delete edge {}".format(round,ed-st,intersection_time,deleted_edges))
                intersection_time = 0
                st = ed
            
            e = self.edge_queue.popleft()
            if self.deleted_node[e[0]] or self.deleted_node[e[1]]:
                continue
            if e not in self.remain_edges:
                continue
            if self.node_degree[e[0]] + self.node_degree[e[1]] >2*self.degree_threshold:
                continue
            if self.isfirst ==False:
                if e[0] not in self.potential_node or e[1] not in self.potential_node:
                    continue
            try:
                pushed_edges.remove(e)
            except:
                pushed_edges.remove((e[1], e[0]))
            if e[0] == e[1]:
                continue
            NG_e0 = self.list_of_set[e[0]]
            
            NG_e1 = self.list_of_set[e[1]]
            
            t1 = time.time()
            is_dominated,dominating_node,NG_e = self.is_dominated_edge_return_node(e)
            #is_dominated_1 = self.is_dominated_edge(e)
            # if is_dominated_1 != is_dominated:
            #     print("error")
            if is_dominated: # e is dominated
                if self.filtration_value_dict[e] < dominating_node:
                    for  n in NG_e:
                        self.filtration_value_dict[(dominating_node,n)] = dominating_node
                        
                        self.filtration_value_dict[(n,dominating_node)] = dominating_node
                    for n in [e[0],e[1]]:
                        self.filtration_value_dict[(n,dominating_node)] = dominating_node
                        self.filtration_value_dict[(dominating_node,n)] = dominating_node
                # self.M[e[0], e[1]] = False
                # self.M[e[1], e[0]] = False
                # self.remain_edges.remove(e)
                # self.remain_edges.remove((e[1], e[0]))
                self.list_of_set[e[0]].remove(e[1])
                self.list_of_set[e[1]].remove(e[0])
                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1
                self.remain_edges.remove(e)
                #delelted_edges_list.append(e)
                deleted_edges += 1

                # filtration_value = self.filtration_value_dict[e]
                # if self.filtration_value_dict[(dominating_node,e[0])] > filtration_value:
                #     self.filtration_value_dict[(dominating_node,e[0])] = filtration_value
                #     self.filtration_value_dict[(e[0],dominating_node)] = filtration_value
                # if self.filtration_value_dict[(dominating_node,e[1])] > filtration_value:
                #     self.filtration_value_dict[(e[1],dominating_node)] = filtration_value
                #     self.filtration_value_dict[(dominating_node,e[1])] = filtration_value

                for neighbor_e in NG_e0:
                    if e[0]!=e[1] and self.deleted_node[neighbor_e]==False and e[0] in self.list_of_set[neighbor_e] and (e[0], neighbor_e) not in pushed_edges and (neighbor_e,e[0])  not in pushed_edges:
                        self.edge_queue.append((e[0], neighbor_e))
                        pushed_edges.add((e[0], neighbor_e))
                        # self.potential_edge.add(neighbor_e)
                for neighbor_e in NG_e1:
                    if e[0] != e[1] and self.deleted_node[neighbor_e]==False  and e[1] in self.list_of_set[neighbor_e]  and (e[1], neighbor_e) not in pushed_edges and (neighbor_e,e[1]) not in pushed_edges:
                        self.edge_queue.append((e[1], neighbor_e))
                        pushed_edges.add((e[1], neighbor_e))
                        # self.potential_edge.add(neighbor_e)
                #self.deleted_edges[e] = max_degree_node
        self.potential_node = set()
        #self.potential_node = []  
        print(count) 
        print(dominate_count) 
        #num_ramain_edge = len(self.get_all_edges()) 
        t1 = time.time()   
        print("end edge collapse,  delete edges, {}, time{} ".format(deleted_edges,t1-t0))
        # if deleted_edges < 10:
        #     self.insert_dominated_edge = True
    

   
    

    def insert_dominated_edges(self):
        delete_node = False
        last_num_nodes = self.num_remain_nodes
        print("insert dominated edges")
        #add_edge_dominate_node = dict()
        for k in self.keep_nodes:
            if self.deleted_node[k]:
                continue
            NG_row = self.list_of_set[k]
            if self.node_degree[k] == 3:
                other_node = [n for n in NG_row if n != k and self.deleted_node[n] == False]
                if other_node[0] not in self.list_of_set[other_node[1]]:
                    
                    NG_e0 = self.list_of_set[other_node[0]]
                    NG_e1 = self.list_of_set[other_node[1]]
                    NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,(other_node[0],other_node[1]))
                    if len(NG_e) == 1:
                        #a = self.node_degree[other_node[0]]
                        #b = self.node_degree[other_node[1]] 
                        delete_node = True
                        self.list_of_set[other_node[0]].add(other_node[1])
                        self.list_of_set[other_node[1]].add(other_node[0]) 
                        self.deleted_node[k] = True
                        
                        #self.inserted_edges[(other_node[0],other_node[1])] = k
                        # if v == 633:
                        #     print("633")
                        #self.need_delete.append(k)
                        self.node_degree[k] = 1
                        
                        self.num_remain_nodes -= 1


                        maximunm = max(self.filtration_value_dict[(other_node[0],k)],self.filtration_value_dict[(other_node[1],k)])
                        #print(e)
                        #print(self.inserted_edges[e])
                        self.filtration_value_dict[(other_node[0], other_node[1])] = maximunm
                        self.filtration_value_dict[(other_node[1], other_node[0])] = maximunm
                        # if torch.cosine_similarity(self.data.x[k],self.data.x[other_node[0]],dim=0) > torch.cosine_similarity(self.data.x[k],self.data.x[other_node[1]],dim=0):
                        #     self.deleted_to_remain[k] = other_node[0] 
                        # else:
                        #     self.deleted_to_remain[k] = other_node[1]

                        
                        self.deleted_to_remain[k] = other_node[0] 
                            
                            #self.label_list[other_node[0]]+=self.label_list[k]
                            
                            #self.node_feature[other_node[0]] = self.node_feature[other_node[0]] + self.node_feature[k]
                            #self.conain_node[other_node[1]] = self.conain_node[other_node[1]] + self.conain_node[k] 
                        
                            #self.label_list[other_node[1]]+=self.label_list[k]
                            
                            #self.node_feature[other_node[1]] = self.node_feature[other_node[1]] + self.node_feature[k]
                            #self.conain_node[other_node[1]] = self.conain_node[other_node[1]] + self.conain_node[k]
                        #self.deleted_to_remain[k] = other_node[0]
                        #print(self.num_reduced_node)
                        
                                #self.node_queue.append(neighbor)
                        
                        # if self.num_remain_nodes <= self.reduction_object:
                        #     self.finish = True
                        #     return
        #self.delayed_delete()
        #self.heruistic_delete = True
        #self.node_queue = deque(list(set(self.node_queue)))
        print("end insert dominated node,  node remain {} delete {} this round ".format(self.num_remain_nodes,last_num_nodes-self.num_remain_nodes)) 
        # self.strong_collapse()
        # self.edge_collapse()
        if delete_node == False:
            self.finish1 = True

        return

    def find_cycles_bfs(self, start_node,second_node):
        min_len = 0
        is_first = True
        cycles = []
        visited = set()
        queue = deque([(second_node, [start_node,second_node])])
        while queue:
            current_node, path = queue.popleft()
            for neighbor in self.M.rows[current_node]:
                if neighbor == current_node:
                    continue
                if neighbor == start_node and len(path) > 3 :
                    if is_first == False and len(path)>min_len-1:
                        return cycles
                    
                    
                    if ind_loop == True:
                        
                        cycles.append(path + [start_node])
                        if is_first == True:
                            is_first = False
                            min_len = len(cycles[0])
                    
                elif neighbor not in path and neighbor <=start_node :
                    ind_loop = True
                    if len(path) >=3:
                        
                        for p in path[1:-1]:
                            if self.M[p,neighbor]:
                                ind_loop = False
                    if ind_loop == True:
                        queue.append((neighbor, path + [neighbor]))
        return cycles


    def insert_dominated_edges_2(self):
        delete_node = False
        st = time.time()
        last_num_nodes = self.num_remain_nodes
        print("insert dominated edges")
        dominate_edge_dict = dict()
        vertex_queue = []
        for k in self.keep_nodes:
            if self.node_degree[k] >2 :
                heapq.heappush(vertex_queue, (self.node_degree[k], k)) 
        degree = dict()
        round =0
        while vertex_queue: 
            deg_k,k = heapq.heappop(vertex_queue)
            if deg_k != self.node_degree[k]:
                continue
            if self.deleted_node[k]:
                continue
            #print(deg_k)
            NG_row = self.list_of_set[k]
            if self.node_degree[k] <2 or self.node_degree[k] > self.degree_threshold2:
                continue
            
            other_node = [n for n in NG_row if n != k and self.deleted_node[n] == False]
            #print(other_node)
            flag = False
            round +=1
            # if round % 10000 == 0:
                
            #     print("round {} insertDE".format(round))
                
            for i in range(len(other_node)):
                NG_i = [n for n in self.list_of_set[other_node[i]] if self.deleted_node[n] == False]
                add_edge = []
                add_edge_np = np.empty((11, 2), dtype=np.int32)
                current_index = 0
                dominate_flag = True # check if other_node[i] can connect to other nodes
                add_edge_dominate_node = dict()
                if self.node_degree[other_node[i]] >= self.node_degree[k] and self.node_degree[k]>3:
                        continue
                for j in range(len(other_node)):
                    if j == i:
                        continue
                    
                    if self.M[other_node[i],other_node[j]] == False:
                        
                        NG_j = [n for n in self.list_of_set[other_node[j]] if self.deleted_node[n] == False]
                        NG_e1 = self.sorted_list_intersection_with_deleted_set(NG_i, NG_j,(other_node[i],other_node[j]))

                        # if len(NG_e1) != 1:
                        #     dominate_flag = False
                                
                        #     break
                        if len(NG_e1) != 1:
                            flag2 = False
                            for a in NG_e1:
                                if self.is_subset_with_deleted_set(NG_e1, self.M.rows[a],a):
                                    flag2 = True
                                    break
                            # if self.is_subset_with_deleted(NG_e1, self.M.rows[k]):
                            #     flag2 = True
                            if flag2 == False:
                                dominate_flag = False

                                break
                            else:
                                
                                #add_edge.append((other_node[i],other_node[j]))
                                add_edge_np[current_index] = [other_node[i], other_node[j]]
                                current_index += 1
                                add_edge_dominate_node[(other_node[i],other_node[j])] = a
                                
                        else:
                            #add_edge.append((other_node[i],other_node[j]))
                            add_edge_np[current_index] = [other_node[i], other_node[j]]
                            current_index += 1
                            add_edge_dominate_node[(other_node[i], other_node[j])] = k
                if dominate_flag == False:
                    continue
                else:
                    flag = True
                    dominating_node = other_node[i]
                if flag == True:
                    break
            if flag==False:
                continue
            else:
                if self.node_degree[k] not in  degree:
                    degree[self.node_degree[k]] = 1
                else:
                    degree[self.node_degree[k]] += 1
                delete_node = True
                add_edge = add_edge_np[:current_index]
                #self.M.rows[651]
                for e in add_edge:
                    self.list_of_set[e[0]].add(e[1])
                    self.list_of_set[e[1]].add(e[0])
                    self.node_degree[e[0]] += 1
                    self.node_degree[e[1]] += 1
                    #self.inserted_edges[e] = add_edge_dominate_node[e]

                #if self.node_degree[k] > 3:
                    # if len(add_edge) +self.node_degree[dominating_node]>self.node_degree[k]  : #or len(add_edge) < self.node_degree[k] - 2
                    #     continue
                    #print(f"degree {self.node_degree[k]}, len add {len(add_edge)} ---------------------------------------------------------------------------------------------")
                    #print(k)
                    # for e in add_edge:
                    #     oth_node = e[1] if e[0] == dominating_node else e[0]
                    #     self.filtration_value_dict[e] = self.filtration_value_dict[(k,oth_node)]
                    #     self.filtration_value_dict[(e[1],e[0])] = self.filtration_value_dict[(k,oth_node)]
                        # if add_edge_dominate_node[(e[0],e[1])] != k:
                        #     print(f" add_edge_dominate_node {add_edge_dominate_node[e]} k {k}")
                        #     if not self.is_subset_with_deleted(NG_e1, self.M.rows[k]):
                        #         print("e is not dominated by k")
                            
                    #if len(add_edge) < self.node_degree[k] - 2:
                ng =list(self.list_of_set[k])
                ng.remove(k)
                #print(self.node_degree[k])
                for i in range(len(ng)):
                    if ng[i] == dominating_node:
                        continue
                    if (dominating_node,ng[i]) in self.filtration_value_dict:
                        if self.filtration_value_dict[(dominating_node,ng[i])] > k:
                            self.filtration_value_dict[(dominating_node,ng[i])] = k
                            self.filtration_value_dict[(ng[i],dominating_node)] = k
                        # maximunm = max(self.filtration_value_dict[(k,oth_node)],self.filtration_value_dict[(dominating_node,k)])
                        # self.filtration_value_dict[e] = maximunm
                        # self.filtration_value_dict[(e[1],e[0])] = maximunm
                    # if self.find_cycles_bfs(k,j) == True:
                    #     continue
                    #self.insert_dominated_node[k] = dominating_node
                
                # else:
                #     for e in add_edge:
                #         #print(e)
                #         #print(self.inserted_edges[e])
                #         if 
                #         self.filtration_value_dict[e] = maximunm
                #         self.filtration_value_dict[(e[1],e[0])] = maximunm
                
                self.deleted_node[k] = True
                #self.need_delete.append(k)
                self.node_degree[k] = 1
               
                
                self.num_remain_nodes -= 1
        
                #if self.label[k] == self.label[other_node[0] ]:
                self.deleted_to_remain[k] = dominating_node 
                    
                    #self.label_list[other_node[0]]+=self.label_list[k]
                    
                    # self.node_feature[other_node[0]] = self.node_feature[other_node[0]] + self.node_feature[k]
                    # self.conain_node[other_node[1]] = self.conain_node[other_node[1]] + self.conain_node[k] 
                #else:
                    #self.deleted_to_remain[k] = other_node[1]
                    
                    #self.label_list[other_node[1]]+=self.label_list[k]
                    
                    # self.node_feature[other_node[1]] = self.node_feature[other_node[1]] + self.node_feature[k]
                    # self.conain_node[other_node[1]] = self.conain_node[other_node[1]] + self.conain_node[k]
                #self.deleted_to_remain[k] = other_node[0]
                #print(self.num_reduced_node)
                #self.edge_queue.append((other_node[0],other_node[1]))
                for neighbor in NG_row:
                    
                    self.node_degree[neighbor] -= 1
                        
                        #self.node_queue.append(neighbor)
                heapq.heappush(vertex_queue, (self.node_degree[dominating_node], dominating_node))
                # if self.num_remain_nodes <= self.reduction_object:
                #     self.finish = True
                #     return
        #self.delayed_delete()
        #print(degree)
        ed = time.time()
        self.heruistic_delete = True
        self.node_queue = deque(list(set(self.node_queue)))
        print("end insert dominated node,  node remain {} delete {} this round , time cost {}".format(self.num_remain_nodes,last_num_nodes-self.num_remain_nodes,ed-st)) 
        # self.strong_collapse()
        # self.edge_collapse()
        if delete_node == False:
            self.finish1 = True

        return

    
    


    def delete_deg2(self):
        original_node = self.num_reduced_node
        print("break 4 ring")
        for k in self.keep_nodes:
            if self.deleted_node[k]:
                continue
            NG_row = self.M.rows[k]
            if len(NG_row) == 3:
                other_node = [n for n in NG_row if n != k]
                if self.M[other_node[0],other_node[1]] == False and self.deleted_node[other_node[0]] == False and self.deleted_node[other_node[1]] == False:
                    self.M[other_node[0], other_node[1]] = True
                    self.M[other_node[1], other_node[0]] = True
                    self.deleted_node[k] = True
                    self.node_degree[k] = 1
                    # self.node_degree[other_node[0]] -= 1
                    # self.node_degree[other_node[1]] -= 1
                    self.need_delete.append(k)
                    self.num_reduced_node += 1

                    if self.label[k] == self.label[other_node[0]]:
                        self.deleted_to_remain[k] = other_node[0] 
                    else:
                        self.deleted_to_remain[k] = other_node[1]
                    #self.deleted_to_remain[k] = other_node[0]
                    self.delayed_delete()
                    #print(self.num_reduced_node)
                    if self.num_reduced_node == self.reduction_object:
                        self.finish = True
                        return
                    for neighbor in self.M.rows[k]:
                        if self.deleted_node[neighbor] == False:
                            self.node_degree[neighbor] -= 1
                            self.node_queue.append(neighbor)
                        self.edge_queue.append((other_node[0],other_node[1]))
                        for e in self.M.rows[other_node[0]]:
                            if e != other_node[1]:
                                self.edge_queue.append((other_node[0],e))
                        for e in self.M.rows[other_node[1]]:
                            if e != other_node[0]:
                                self.edge_queue.append((other_node[1],e))
                       #NG_e0 = self.M.rows[k]

                    # self.strong_collapse()
                    # self.edge_collapse()
                    if self.num_reduced_node == self.reduction_object:
                        self.finish = True
                        return
        self.heruistic_delete = False
        
        if original_node == self.num_reduced_node:
            self.heruistic_delete_deg3 = True
        return
    
    def delete_deg3(self):
        print("delete degree 3")
        delete_flag = False
        for k in self.keep_nodes:
            if self.deleted_node[k]:
                continue
           
            NG_row = self.M.rows[k]
            if len(NG_row) == 4:
                other_node = [n for n in NG_row if n != k]
                self.deleted_node[k] = True
                self.need_delete.append(k)
                delete_flag = True
                self.num_reduced_node += 1

            
                
                for i in other_node:
                    if self.label[k] == self.label[i]:
                        self.deleted_to_remain[k] = i
                        break
                    self.deleted_to_remain[k] = i
                for i in other_node:
                    if i != k or i!= self.deleted_to_remain[k]:
                        if  self.M[i,self.deleted_to_remain[k]] == False:
                            self.M[i,self.deleted_to_remain[k]] = True
                            self.M[self.deleted_to_remain[k],i] = True
                            self.node_degree[self.deleted_to_remain[k]] += 1
                        
                        
                self.delayed_delete()
                self.node_degree[k] = 1

                if self.num_reduced_node == self.reduction_object:
                    self.finish = True
                    return 
        
        if delete_flag == False:
            self.finish = True
        
        return

    
    def heruistic_delete_edge(self):
        deleted_edge_num = 0
        upper_triangle = triu(self.M, k=1)
        # 获取所有的非零元素坐标（边）
        rows, cols = upper_triangle.nonzero()
        edges = list(zip(rows, cols))
        delete_flag = False
        for e in edges:
        #ramdom delete edge
            if self.node_degree[e[0]] == 3 or self.node_degree[e[1]] == 3:
                delete_flag = True
                self.M[e[0],e[1]] = False
                self.M[e[1],e[0]] = False

                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1

                self.node_queue.append(e[0])
                self.node_queue.append(e[1])
                deleted_edge_num += 1

                
                # NG_e0 = self.M.rows[e[0]]
                # NG_e1 = self.M.rows[e[1]]

                # for neighbor_e in NG_e0:
                #     #
                #     if neighbor_e == e[0]:
                #         continue
                #     if self.M[e[0],neighbor_e] == True :
                        
                #         edge_queue.append((random_edge[0],neighbor_e))
                #         pushed_edges.add((random_edge[0],neighbor_e))
                    
                # if deleted_node[neighbor_e] == False:
                #     node_queue.append(neighbor_e)
        self.heruistic_delete = False
        if delete_flag == False:
            self.finish = True
        
        return
 

            

    def delayed_delete(self):
        while self.need_delete:
            i = self.need_delete.popleft()
            for j in self.M.getrow(i).nonzero()[1]:
                if i != j:
                    self.M[i, j] = False
                    self.M[j, i] = False
    

    def find_root(self, map, elem):
    # 找到最终指向自己的元素
        while map[elem] != elem:
            elem = map[elem]
        return elem

    # def make_coarsened_graph(self):
    #     old_to_new = {}
        
        
    #     edge_list =np.empty((self.M.shape[0]*20, 2), dtype=np.int32)
    #     new_node_num = sum(self.deleted_node == False)
    #     #res.new_x = torch.zeros(new_node_num, self.data.x.shape[1])
    #     index = 0

    #     # 映射未删除节点到新的节点编号
    #     for i, v in enumerate(self.deleted_node):
    #         if v == False :
    #             old_to_new[i] = index
    #             index += 1

     
    #     index = 0
    #     filtration_value_dict_new = dict()
    #     for i in range(self.M.shape[0]):
    #         if self.deleted_node[i] == False:
    #             for j in self.M.rows[i]:
    #                 if i != j and self.deleted_node[j] == False:
    #                     edge_list[index]=[old_to_new[i], old_to_new[j]] 
    #                     index += 1  
    #                     filtration_value_dict_new[(old_to_new[i], old_to_new[j])] = self.filtration_value_dict[(i,j)]
    #                     filtration_value_dict_new[(old_to_new[j], old_to_new[i])] = self.filtration_value_dict[(i,j)]
        
    #     edge_list = edge_list[:index]
        
    #     return filtration_value_dict_new,edge_list,new_node_num
    def make_coarsened_graph(self):
        # 向量化 old_to_new 映射构建
        mask = ~self.deleted_node
        new_node_num = np.count_nonzero(mask)
        old_to_new_arr = np.full_like(self.deleted_node, fill_value=-1, dtype=np.int32)
        old_to_new_arr[mask] = np.arange(new_node_num, dtype=np.int32)

        edge_list = np.empty((len(self.list_of_set)* self.edge_num, 2), dtype=np.int32)
        filtration_value_dict_new = {}
        index = 0

        # 主循环优化：只遍历保留节点
        for i in np.flatnonzero(mask):
            row = self.list_of_set[i]
            for j in row:
                if i != j and mask[j]:
                    ni, nj = old_to_new_arr[i], old_to_new_arr[j]
                    edge_list[index] = [ni, nj]
                    filtration_value = self.filtration_value_dict.get((i, j), 0.0)
                    filtration_value_dict_new[(ni, nj)] = filtration_value
                    filtration_value_dict_new[(nj, ni)] = filtration_value
                    index += 1

        edge_list = edge_list[:index]
        return filtration_value_dict_new, edge_list, new_node_num
    
    def get_all_edges(self):
        upper_triangle = triu(self.M, k=1)
        rows, cols = upper_triangle.nonzero()
        return list(zip(rows, cols))

    def check_degree(self):
        for i in range(self.M.shape[0]):
            if self.node_degree[i] != len(self.M.rows[i]) and self.deleted_node[i] == False:
                print("degree error")
    def get_adjacency_matrix(self):
        return self.M
    def num_remain_nodes(self):
        return sum(self.deleted_node == False)
    def compute_betti_numbers(self):
        filtered_adj_matrix = self.M[~self.deleted_node, :]  # 删除行
        filtered_adj_matrix = filtered_adj_matrix[:, ~self.deleted_node]
        
        
        return compute_betti_numbers(filtered_adj_matrix)
    
    def run_strong_edge_collapse(self):

        #tracemalloc.start()
        st = time.time()
        self.node_queue = deque(self.keep_nodes)
        # while not self.finish :
        #     self.strong_collapse()
        #     if self.insert_dominated_edge == True:
        #         break
        #     else:
        #         self.edge_collapse()
        #         self.isfirst = False
        #         if self.insert_dominated_edge == True:
        #             break
        #self.edge_collapse()
        #self.insert_dominated_edges()
        self.strong_collapse()
        if self.degree_threshold2 > 0:
            
            self.edge_collapse()
            if self.degree_threshold2 == 2:
                    self.insert_dominated_edges()
            else:
                self.insert_dominated_edges_2()

        # if self.insert_dominated_edge==True:
        #     if self.degree_threshold2 == 2:
        #         self.insert_dominated_edges()
        #     else:
        #         self.insert_dominated_edges_2()
                
        #self.revise_inserted_edge_2_test()

        t1 = time.time()
        new_dict,new_edge,num_new_node = self.make_coarsened_graph()
        ed = time.time()
        print(f"make graph collapse time {t1-ed}")

        #current, peak = tracemalloc.get_traced_memory()
    
        #peak = peak / 1024 ** 2
        #tracemalloc.stop()
        return new_dict,new_edge,num_new_node,0

    def run_edge_collapse(self):

        #tracemalloc.start()
        st = time.time()
        self.node_queue = deque(self.keep_nodes)
        # while not self.finish :
        #     self.strong_collapse()
        #     if self.insert_dominated_edge == True:
        #         break
        #     else:
        #         self.edge_collapse()
        #         self.isfirst = False
        #         if self.insert_dominated_edge == True:
        #             break
        self.strong_collapse()
       

        # if self.insert_dominated_edge==True:
        #     if self.degree_threshold2 == 2:
        #         self.insert_dominated_edges()
        #     else:
        #         self.insert_dominated_edges_2()
                
        #self.revise_inserted_edge_2_test()

        t1 = time.time()
        new_dict,new_edge,num_new_node = self.make_coarsened_graph()
        ed = time.time()
        print(f"make graph collapse time {t1-ed}")

        #current, peak = tracemalloc.get_traced_memory()
    
        #peak = peak / 1024 ** 2
        #tracemalloc.stop()
        return new_dict,new_edge,num_new_node,0


        
        
def is_symmetric_lil(sparse_matrix):
    """
    判断一个 LIL 格式的稀疏矩阵是否是对称的。
    :param sparse_matrix: LIL 格式的稀疏矩阵
    :return: 如果矩阵是对称的返回 True，否则返回 False
    """
   
    
    # 检查是否是方阵
    if sparse_matrix.shape[0] != sparse_matrix.shape[1]:
        return False

    # 转换为 CSR 格式进行高效比较
    sparse_matrix_csr = sparse_matrix.tocsr()
    transpose_csr = sparse_matrix_csr.transpose()

    # 比较矩阵与其转置
    return (sparse_matrix_csr != transpose_csr).nnz == 0

#original_reuslt,old_map = np.load(f'./Reduced_Node_Data/{dataset_name}_0.30_split_All_Simplex_1.npy', allow_pickle=True)
if __name__ == "__main__":

    dataset_name = "ogbn-arxiv"

    if dataset_name == "Cora":
        data = torch.load('/home/wuxiang/GEC-main/Cora/processed/data.pt')[0]
        edges = data.edge_index
        label = data.y
    elif dataset_name == "dblp":
        dataset = CitationFull(root='/home/wuxiang/GEC-main/dataset', name=dataset_name)
        data = dataset[0]
        edges = data.edge_index
    elif dataset_name == "Citeseer":
        dataset_path = './dataset/citeseer'
        dataset =  Planetoid(root='./dataset/Citeseer', name='Citeseer')
        data = dataset[0]
        edges = data.edge_index
        label = data.y
    elif dataset_name == "pubmed":
        dataset = Planetoid(root='./dataset/pubmed', name='pubmed')
        data = dataset[0]
        edges = data.edge_index
        label = data.y
    elif dataset_name == "ogbn-arxiv":
        dataset_path = './dataset/arxiv'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/wuxiang/strong_collapse/dataset/arxiv')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-arxiv')
        data = dataset[0]
        edges = data.edge_index
        reversed_edges = edges[[1, 0]]
        edges = torch.cat([edges, reversed_edges], dim=1)
    
    # model = torch.load(f'./model_dir/mlp_model_{dataset_name}.pt')
    # model.eval()
    indices = []
    if dataset_name == "ogbn-arxiv":
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
        data.y= data.y.view(-1)
        label = data.y
    
    num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
    index_train = (data.train_mask == 1).nonzero().view(-1)
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        tensor_isin = torch.isin(index, index_train)
        index = index[tensor_isin]
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    keep_index = torch.cat([i[:int(i.size()[0] * 1.0)] for i in indices], dim=0)
    keep_mask = index_to_mask(keep_index, size=data.num_nodes)
    node_mask = np.zeros(data.num_nodes)

    if dataset_name == "ogbn-arxiv":
        keep_nodes = [i for i in range(data.num_nodes) ]
    else:
        keep_nodes = find_component(data)
    edges = np.array(edges)
    #edges = edges.T
    #len(graph.nodes())
    num_nodes = max(edges[0].max(), edges[1].max()) + 1 
    
    adj_matrix_sparse = build_sparse_adjacency_matrix(edges,num_nodes)

    sparse_identity_matrix = eye(num_nodes, format='coo')
    print(f"density {len(edges[0])/num_nodes}")
    adj = adj_matrix_sparse + sparse_identity_matrix
    adj = adj.tolil()
    print(adj.shape[0])
    adj = adj.astype(bool)
 

    
    collapse = CoreAlgorithm(adj,data,keep_nodes,0.1,label,dataset_name,save = True)
    #collapse.run_algorithm_relaxed_strong_collapse()
    #collapse.draw_graph("collapsed_graph_figure_final.png")
    print("remain nodes :{}".format(collapse.num_remain_nodes))


#     b1,b2,_ = collapse.compute_betti_numbers()
#     print(f"Betti 1 (环的数量): {b1}")
#     print(f"Betti 2 (空腔的数量): {b2}")
# # st = gd.SimplexTree()
# edges = collapse.get_all_edges()
# for i,e in enumerate(edges):
#     st.insert([e[0],e[1]], filtration=1)
# result_str = 'Rips complex is of dimension ' + repr(st.dimension()) + ' - ' + \
# repr(st.num_simplices()) + ' simplices - ' + \
# repr(st.num_vertices()) + ' vertices.'
# print(result_str)
# fmt = '%s -> %.2f'
# for filtered_value in st.get_filtration():
#     print(fmt % tuple(filtered_value))
# st.compute_persistence()

# # 提取持久性图
# diag = st.persistence_intervals_in_dimension(1)
# print("持久性图:", diag)
#_,new_map = np.load(f'./Reduced_Node_Data/{dataset_name}_0.30_split_All_Simplex_1.npy', allow_pickle=True)

# for key,values in old_map.items():
#     if old_map[i] != new_map[i]:
#         print("error")
#         break


#num_collpsed_edges = np.count_nonzero(collapsed_M) 
# filtered_adj_matrix = collapsed_M[~deleted_node, :]  # 删除行
# filtered_adj_matrix = filtered_adj_matrix[:, ~deleted_node]
# #print(num_collpsed_edges/2)

# M = filtered_adj_matrix.copy() - np.eye(filtered_adj_matrix.shape[0])
# vis_M = nx.from_numpy_array(M)
# plt.figure(figsize=(10, 10))
# nx.draw(vis_M, with_labels=True, node_color='lightblue', node_size=100, font_size=16)
# plt.savefig("collapsed_graph_figure_final.png", format="png")




# M = collapsed_M.copy() - np.eye(collapsed_M.shape[0])
# vis_M = nx.from_numpy_array(M)
# plt.figure(figsize=(10, 10))
# nx.draw(vis_M, with_labels=True, node_color='lightblue', node_size=100, font_size=16)
# plt.savefig("edge_collapsed_graph_figure_1.png", format="png")
