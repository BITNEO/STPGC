import torch
from torch_geometric.datasets import Planetoid,Reddit
import networkx as nx
import numpy as np
from collections import deque
import torch_geometric.transforms as T
from scipy.sparse import eye, coo_matrix,triu
from torch_geometric.datasets import CitationFull,Coauthor
import argparse
import time
import copy
from pybloom_live import BloomFilter
#import gudhi as gd
#from compute_betti_number import compute_betti_numbers
#from gudhi.representations import WassersteinDistance
# 加载 Cora 数据集
from PyGdataset import PygNodePropPredDataset,Evaluator
#from compute_betti_number import compute_diagram_ripser
# edge collapse of a flag complex
# 只取第一个图数据


import random
import heapq

from numba import njit

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
    for i in range(len(data['x'])):
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
    for i in range(len(data['edge_index'][0])):
        n1n1 = int(data['edge_index'][0, i])
        n2n2 = int(data['edge_index'][1, i])
        if n1n1 != n2n2:
            graph.add_edge(n1n1, n2n2)
        # if int(data['edge_index'][0, i]) == 0 or int(data['edge_index'][1, i]) == 0:
        #     print("???")
        # n[int(data['edge_index'][0, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
        # n[int(data['edge_index'][1, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
    # print(graph.nodes())
    # print(graph.edges())
    cnt_cluster_node = 0
    num_nodes = data['x'].size()[0]
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


def sorted_list_intersection_with_deleted( list1, list2):
    i, j = 0, 0
    intersection = np.empty(len(list1),dtype=np.int32)
    count = 0 
    n, m = len(list1), len(list2)
    # 使用双指针遍历两个列表
    while i < n and j < m:
        # 跳过 list1 中被标记删除的元素
        
        # 如果其中一个列表已经遍历完，结束循环
       
        
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
    def __init__(self, M, data,keep_nodes, reduction_ratio,node_label,args,save = True):
        self.data = data
        self.M = M
        #构建list of set
        self.list_of_set = []
        for i in range(M.shape[0]):
            self.list_of_set.append(set(M.rows[i]))
        # self.adj_2_hop = M @ M
        #self.adj_2_hop = self.adj_2_hop.tolil()
        self.args = args
        self.num_delete_hetero = args.del_edge
        self.insert_dominated_edge_degree_limit = args.deg2
        self.degree_threshold = args.deg1

        self.keep_nodes = keep_nodes
        self.reduction_ratio = reduction_ratio
        self.reduction_object = int(self.M.shape[0] * (reduction_ratio))
        self.deleted_to_remain = {v: v for v in keep_nodes}
        self.num_remain_nodes = len(self.keep_nodes)
        
        #self.deleted_node = np.array([True for i in range(M.shape[0])])
        self.deleted_node = np.full(M.shape[0], fill_value=True, dtype=bool)
        #draw_graph(adj,-1)
        self.dataname = args.dataname
        for i in keep_nodes:
            self.deleted_node[i] = False
        self.degree_threshold_limit = 0.01*self.M.shape[0]
                
        self.edges = set()
        self.node_degree = np.array([M[i].nnz for i in range(M.shape[0])])
        self.need_delete = deque([i for i in range(M.shape[0]) if self.deleted_node[i]])
        #self.node_queue = deque(keep_nodes)
        self.edge_queue = deque()
        self.round = 0
        self.finish = False
        self.insert_dominated_edge = False
        self.keep_mask = ~(copy.deepcopy(self.deleted_node))

        self.heruistic_delete = False
        self.label = node_label.numpy()
        self.heruistic_delete_deg3 = False
        self.save = save
        self.remain_edges = set()

        #self.remain_edges = self.get_all_edges_set()
        edges = self.get_all_edges()
        for e in edges:    
            self.remain_edges.add(e)
      
       
        self.old_to_new = dict()
        self.new_to_old = dict()
        try:
            #self.node_feature = data['x']
            self.test_mask = data.test_mask
        except:
            self.test_mask = [False for i in range(len(node_label))]
            #self.node_feature = torch.zeros((len(node_label),1))
            pass
        self.test_mask = self.test_mask.numpy()
        self.label_list = [[node_label[i]] if self.test_mask[i] == False else [] for i in range(len(node_label))]
        self.strong_relaxed = False
        
        self.finish1 = False
        #self.filtration_value_dict = dict()
        self.finish2 = False
        #self.conain_node = [1 for i in range(len(node_label))]

        #self.edge_intersection = dict()
        #self.dirty_edge = dict()
        #self.dirty_node = [False for i in range(M.shape[0])]
        #self.edge_intersection_max_node = dict()
        self.potential_node = set()
        self.potential_edge = set()
        self.isfirst = True
        self.finish_reduction = False
        self.strong_tolerance = 0
        
        self.node_bloom_filters = {}
        # for n in keep_nodes:
        #     bf = BloomFilter(capacity=self.node_degree[n], error_rate=0.01)
        #     for neighbor in self.M.rows[n]:
        #         bf.add(neighbor)
        #     self.node_bloom_filters[n] = bf        
        #self.initilize_bloom_filter()

    def initilize_bloom_filter(self):
        edges = self.get_all_edges()
        for e in edges:
            for node in [e[0], e[1]]:
                
                if node not in self.filters:
                    self.filters[node] = CountingBloomFilter(1000, 1)
                self.filters[node].add(str(e[0] if node == e[1] else e[1]))
        
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
        
        for e in self.remain_edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1
        
        print(f"homo edge: {homo/count}")

        
        delete = 0
        new_set = self.remain_edges.copy()
        for e in self.remain_edges:
            
            if  node_label[e[0]] != node_label[e[1]] and  node_label[e[0]] != -1 and node_label[e[1]] != -1: 
                
                if delete >= obj_num:
                    break
                self.list_of_set[e[0]].remove(e[1]) 
                self.list_of_set[e[1]].remove(e[0]) 
                new_set.remove(e)
                delete +=1 
                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1 
                # for neighbor in self.M.rows[e[0]]:
                #     self.potential_edge.add(neighbor)
                # for neighbor in self.M.rows[e[1]]:
                #     self.potential_edge.add(neighbor)
        self.remain_edges=new_set        
        
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
        
        for e in  self.remain_edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1
        
        print(f"homo edge: {homo/count}")
        self.homo_ratio = homo/count
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

    def swap_mis_label(self,old_to_new, new_to_old,adj_matrix_sparse):
        supernode_label = dict()
        label_list = dict()
        total_label_count = 0
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]
            total_label_count += len(label_list[key])
        print(f"total label {total_label_count}")
        for key,value in label_list.items():
            if len(value) == 0:
                supernode_label[key] = 0
                continue
            values, counts = np.unique(value, return_counts=True)

            # 找出出现次数最多的元素
            max_label = values[np.argmax(counts)]
            supernode_label[key] = max_label

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1
        
        print(f"mis percentage {mis_label_count/total_label_count}")

        for key,value in new_to_old.items():
            
            value_filtered = [v for v in value if not self.test_mask[v]]
            for v in value_filtered:
                if self.label[v] != supernode_label[key]:
                    for neighbor in adj_matrix_sparse.rows[key]:
                        if supernode_label[neighbor] == self.label[v]:
                            old_to_new[v] = neighbor
                            new_to_old[neighbor].append(v)
                            new_to_old[key].remove(v)
                            break
        label_list = dict()
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1
        print(f"revised mis percentage {mis_label_count/total_label_count}")            
        return old_to_new,new_to_old,label_list,supernode_label
    
    def swap_mis_label_any_node(self,old_to_new, new_to_old,adj_matrix_sparse):
        supernode_label = dict()
        label_list = dict()
        total_label_count = 0
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]
            total_label_count += len(label_list[key])
        print(f"total label {total_label_count}")
        for key,value in label_list.items():
            if len(value) == 0:
                supernode_label[key] = 0
                continue
            values, counts = np.unique(value, return_counts=True)

            # 找出出现次数最多的元素
            max_label = values[np.argmax(counts)]
            supernode_label[key] = max_label

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1
        
        print(f"mis percentage {mis_label_count/total_label_count}")

        label_to_node = {}
        for k,value in supernode_label.items():
            if value not in label_to_node:
                label_to_node[value] =  [k]
            else:
                label_to_node[value].append(k)
        
        for key,value in new_to_old.items():
            
            value_filtered = [v for v in value if not self.test_mask[v]]
            for v in value_filtered:
                if self.label[v] != supernode_label[key]:
                    random_supernode = np.random.choice(label_to_node[self.label[v]])
                    old_to_new[v] = random_supernode
                    new_to_old[random_supernode].append(v)
                    new_to_old[key].remove(v)
                    break
        label_list = dict()
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1
        print(f"revised mis percentage {mis_label_count/total_label_count}")            
        return old_to_new,new_to_old,label_list,supernode_label

    
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

    # def draw_graph(self,save_name): 
    #     edges = self.get_all_edges()
    #     print(len(edges))
        
    #     vis_M = nx.Graph()
    #     vis_M.add_edges_from(edges)
    #     dict(vis_M.degree())
    #     plt.figure(figsize=(30, 30))
    #     nx.draw(vis_M, with_labels=True, node_color='lightblue', node_size=20, font_size=10)
    #     plt.savefig(save_name)



    #@numba.jit(nopython=True) 
    def sorted_list_intersection_with_deleted(self, list1, list2):
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
        # while i < n and j < m:
        #     # 跳过 list1 中被标记删除的元素
        #     while i < n and self.deleted_node[list1[i]]:
        #         i += 1
            
        #     # 跳过 list2 中被标记删除的元素
        #     while j < m and self.deleted_node[list2[j]]:
        #         j += 1
            
        #     # 如果其中一个列表已经遍历完，结束循环
        #     if i >= n or j >= m:
        #         break
            
        #     if list1[i] == list2[j]:
                
        #         intersection[count] = list1[i]
        #         count += 1
        #         i += 1
        #         j += 1
        #     elif list1[i] < list2[j]:
        #         i += 1  # 移动 list1 的指针
        #     else:
        #         j += 1  # 移动 list2 的指针
        
        # return intersection[:count]
    
    def is_dominated_edge(self,e):
        NG_e0 = self.M.rows[e[0]]
            
        NG_e1 = self.M.rows[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted(NG_e0, NG_e1)
        NG_e = [n for n in NG_e if n not in e]
        if len(NG_e) == 0:
            return False
        if len(NG_e) == 1:
            return True
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False
        for w in NG_e:
            is_dominated=True
            #NG_w = self.M.rows[w]
            for n in NG_e:
                if not self.M[w,n]:
                    is_dominated = False
                    break
                
            if is_dominated == False:
                continue
            else:
                is_dominated = True
                return True
        if is_dominated == False:
            return False
    
        
    def is_dominated_edge_set(self,e):
        NG_e0 = self.list_of_set[e[0]]
            
        NG_e1 = self.list_of_set[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,e)
        NG_e = [n for n in NG_e if n not in e ]

        if len(NG_e) == 0:
            return False
        if len(NG_e) == 1:
            return True
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False
        for w in NG_e:
            is_dominated=True
            #NG_w = self.M.rows[w]
            for n in NG_e:
                # if (w,n) not in self.remain_edges and (n,w) not in self.remain_edges:
                if w==n:
                    continue
                if w not in self.list_of_set[n]:
                    is_dominated = False
                    break
                
            if is_dominated == False:
                continue
            else:
                is_dominated = True
                return True
        if is_dominated == False:
            return False
    

    def is_dominated_edge_bf(self,e):
        NG_e0 = self.M.rows[e[0]]
            
        NG_e1 = self.M.rows[e[1]]
        NG_e = sorted_list_intersection_with_deleted(NG_e0, NG_e1)
        NG_e = [n for n in NG_e if (n not in e and self.deleted_node[n] == False)]
        if len(NG_e) == 0:
            return False
        if len(NG_e) == 1:
            return True
        
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False
        for w in NG_e:
            is_dominated=True
            #NG_w = self.M.rows[w]
            for n in NG_e:
                if n not in self.node_bloom_filters[w]:
                    is_dominated = False
                    break
                
            if is_dominated == False:
                continue
            else:
                #for n in NG_e:
                    # if not self.M[w,n]:
                    #     is_dominated = False
                    #     break
                for n in NG_e:
                    if (w,n) not in self.remain_edges:
                        is_dominated = False
                        break
                    
                if is_dominated == False:
                    continue
                else:
                    is_dominated = True
                    return True
        if is_dominated == False:
            return False
                    
            
        
        
    def is_dominated_edge_return_node(self,e):
        NG_e0 = self.list_of_set[e[0]]
            
        NG_e1 = self.list_of_set[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,e)
        #NG_e = [n for n in NG_e if (n not in e and self.deleted_node[n] == False)]
        NG_e = [n for n in NG_e if n not in e ]
        if len(NG_e) == 0:
            return False,0
        if len(NG_e) == 1:
            return True,NG_e[0]
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False,0
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
                return True,w
        if is_dominated == False:
            return False,0

    # def is_dominated_edge(self,e):
    #     NG_e0 = self.M.rows[e[0]]
            
    #     NG_e1 = self.M.rows[e[1]]
    #     NG_e = self.sorted_list_intersection_with_deleted(NG_e0, NG_e1)
    #     #NG_e = [n for n in NG_e if n not in e]
    #     e_np = np.array(e)
    #     NG_e = np.setdiff1d(NG_e, e_np, assume_unique=True)
    #     if NG_e.size == 0:
    #         return False
    #     if NG_e.size == 1:
    #         return True
    #     is_dominated = True
    #     node_degree_np = np.array([self.node_degree[n] for n in NG_e])
    #     max_degree_idx = np.argmax(node_degree_np)
    #     max_degree_node = NG_e[max_degree_idx]
    #     if self.node_degree[max_degree_node] < NG_e.size:
    #         return False
    #     for w in NG_e:
    #         is_dominated=True
    #         #NG_w = self.M.rows[w]
    #         for n in NG_e:
    #             if not self.M[w,n]:
    #                 is_dominated = False
    #                 break
                
    #         if is_dominated == False:
    #             continue
    #         else:
    #             is_dominated = True
    #             return True
    #     if is_dominated == False:
    #         return False
    
        
    #@numba.jit(nopython=True) 
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
    
    def is_subset_with_deleted_Tolerance(self, sorted_list1, sorted_list2,w,tolerance):
        #flag = True 
        count = 0
        for n_list1 in sorted_list1:
            if self.deleted_node[n_list1] == False and n_list1 not in self.node_bloom_filters[w]:
                count += 1
                if count > tolerance:
                    return False
                
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
        # if flag == False:
        #     print("error")

        return True

    def is_subset_with_deleted_set_Tolerance(self, sorted_list1, sorted_list2,w,tolerance):
        #flag = True 
        count = 0
        for n_list1 in sorted_list1:
            if self.deleted_node[n_list1] == False and n_list1 not in self.list_of_set[w]:
                count += 1
                if count > tolerance:
                    return False
        return True
                
        # i, j = 0, 0
        # n, m = len(sorted_list1), len(sorted_list2)
        # mismatch_count = 0  # 记录不匹配的元素数量
        

        # while i < n:
        #     # 找到 sorted_list1 中下一个有效元素
        #     while i < n and self.deleted_node[sorted_list1[i]]:
        #         i += 1

        #     # 找到 sorted_list2 中下一个有效元素
        #     while j < m and self.deleted_node[sorted_list2[j]]:
        #         j += 1

        #     if i >= n:  # list1 已遍历完
        #         return True

        #     if j >= m:  # list2 已遍历完，但 list1 还没结束
        #         mismatch_count += 1
        #         if mismatch_count > tolerance:
        #             return False
        #         i += 1
        #         continue

        #     if sorted_list1[i] < sorted_list2[j]:
        #         mismatch_count += 1
        #         if mismatch_count > tolerance:
        #             return False
        #         i += 1
        #     elif sorted_list1[i] == sorted_list2[j]:
        #         i += 1
        #     j += 1  # 始终移动 list2 的指针

        #     # 如果不匹配的元素超过 1 个，返回 False
        # # if flag == False:
        # #     print("error")

        # return True

    
    
    def strong_collapse(self):
        st = time.time()
        #node_queue = deque(keep_nodes)
        
        pushed_nodes = set(list(self.node_queue))
        #print(2034 in pushed_nodes)
        
        collapse_count = dict()
        st_time = time.time()

        set_degree_flag = False

        last_num_nodes = self.num_remain_nodes
        print(f"round {self.round} strong collapse, node_num {len(self.node_queue)}")
        while self.node_queue:
            self.round += 1
            if self.round % 10000 == 0:
                ed = time.time()
                print("round {} strong collapse: {}".format(round,ed-st))
                st = ed
                print(len(self.node_queue))

            v = self.node_queue.popleft()
            pushed_nodes.remove(v)
            if self.isfirst == False and self.heruistic_delete == False and v not  in self.potential_edge:
                continue
            if  self.node_degree[v]>self.degree_threshold:
                continue

            
            
            if self.deleted_node[v]:
                continue
            
            NG_v = self.list_of_set[v]

            
            # node_label = [a for a in NG_v if self.label[a] == self.label[v] ]

            same_label_nodes = []
            other_nodes = []

            # 单次遍历完成分组
            for node in NG_v:
                if (not self.test_mask[node]) and (self.label[node] == self.label[v]):
                    same_label_nodes.append(node)
                else:
                    other_nodes.append(node)

            sorted_other_node = same_label_nodes + other_nodes
            
            # label_v = self.label[v]
            # label_NG_v = self.label[NG_v]
            # score = abs(label_NG_v - label_v)
            # sorted_NG_v = [x for x, y in sorted(zip(NG_v, score), key=lambda pair: pair[1])]
            # if self.num_remain_nodes <= self.reduction_object:
            #     self.finish_reduction = True
            #     return 
            for w in sorted_other_node:
                if w == v or self.deleted_node[w] or (self.node_degree[w]+self.strong_tolerance) < self.node_degree[v]:
                    continue
                NG_w = self.list_of_set[w]
                #if len(self.sorted_list_intersection_with_deleted(NG_v, NG_w)) == self.node_degree[v]:
                
                if self.is_subset_with_deleted_set_Tolerance(NG_v, NG_w,w,self.strong_tolerance):
                    
                    self.deleted_node[v] = True
                    #self.need_delete.append(v)
                    self.node_degree[v] = 1
                    self.deleted_to_remain[v] = w
                    
                    # if not set_degree_flag:
                    #     if self.node_degree[v] not in collapse_count:
                    #         collapse_count[self.node_degree[v]] = 1
                    #     else:
                    #         collapse_count[self.node_degree[v]] += 1

                    
                    self.label_list[w]+=self.label_list[v]
                   
                    #self.node_feature[w] = self.node_feature[w] + self.node_feature[v]
                    #self.conain_node[w] = self.conain_node[w] + self.conain_node[v] 
                    self.num_remain_nodes -= 1
                    if self.num_remain_nodes <= self.reduction_object:
                        self.finish_reduction = True
                        return 

                    for neighbor in NG_v:
                        if not self.deleted_node[neighbor]:
                            self.node_degree[neighbor] -= 1
                            
                            if self.isfirst == False:
                                self.potential_node.add(neighbor)
                            if neighbor not in pushed_nodes:
                                pushed_nodes.add(neighbor)
                                self.node_queue.append(neighbor)
                    break
               
        #self.delayed_delete()
        self.potential_node_ = set()
        self.potential_edge = set()
        num_remain_nodes = self.num_remain_nodes
        ed_time = time.time()
        print(f"round {self.round} strong collapse: {ed_time-st_time}s, reduce_nodes total {len((self.keep_nodes))-self.num_remain_nodes}, reduce nodes this round {last_num_nodes-num_remain_nodes} remain_nodes {self.num_remain_nodes}")
        if last_num_nodes - num_remain_nodes == 0:
            self.insert_dominated_edge = True
            
        
        if self.heruistic_delete:
            if last_num_nodes - num_remain_nodes < self.degree_threshold_limit:
                self.strong_tolerance+=1
                self.degree_threshold += 1
                print(f"strong tolerance +1 {self.strong_tolerance}")
            return
            
        else:
            self.finish	 = False
            return 
    
        
    
    
    def intersect_simd(self,a, b):
        a_set = np.array(a)
        b_set = np.array(b)
        return np.intersect1d(a_set, b_set)

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
            is_dominated = self.is_dominated_edge_set(e)
            #is_dominated_1 = self.is_dominated_edge(e)
            # if is_dominated_1 != is_dominated:
            #     print("error")
            if is_dominated: # e is dominated
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
                for neighbor_e in NG_e0:
                    if e[0]!=e[1] and self.deleted_node[neighbor_e]==False and e[0] in self.list_of_set[neighbor_e] and (e[0], neighbor_e) not in pushed_edges and (neighbor_e,e[0])  not in pushed_edges:
                        self.edge_queue.append((e[0], neighbor_e))
                        pushed_edges.add((e[0], neighbor_e))
                        self.potential_edge.add(neighbor_e)
                for neighbor_e in NG_e1:
                    if e[0] != e[1] and self.deleted_node[neighbor_e]==False  and e[1] in self.list_of_set[neighbor_e]  and (e[1], neighbor_e) not in pushed_edges and (neighbor_e,e[1]) not in pushed_edges:
                        self.edge_queue.append((e[1], neighbor_e))
                        pushed_edges.add((e[1], neighbor_e))
                        self.potential_edge.add(neighbor_e)
                #self.deleted_edges[e] = max_degree_node
        self.potential_node = set()
        #self.potential_node = []  
        print(count) 
        print(dominate_count) 
        #num_ramain_edge = len(self.get_all_edges()) 
        t1 = time.time()   
        print("end edge collapse,  delete edges, {}, time{} ".format(deleted_edges,t1-t0))
        #print(delelted_edges_list[:100])
    


    

    

    def insert_dominated_edges_2(self):
        delete_node = False
        st = time.time()
        last_num_nodes = self.num_remain_nodes
        print("insert dominated edges")
        vertex_queue = []
        check_num = 0
        reduce_num = 0
        has_checked = set()
        for k in self.keep_nodes:
            if self.node_degree[k] >2 :
                heapq.heappush(vertex_queue, (self.node_degree[k], k)) 
        degree = dict()
        round =0
        while vertex_queue: 
            deg_k,k = heapq.heappop(vertex_queue)
            if deg_k != self.node_degree[k] or k  in has_checked or self.deleted_node[k]:
                continue
            
            
            
            if self.node_degree[k] <=2 or self.node_degree[k] > self.insert_dominated_edge_degree_limit:
                continue
            NG_row = self.list_of_set[k]
            check_num +=1
            other_node = [n for n in NG_row if n != k and self.deleted_node[n] == False]
            
            other_node = np.array(other_node, dtype=np.int32)  # 如果还不是 NumPy 数组的话
            
# 获取标签  
            labels = self.label[other_node]
            target_label = self.label[k]
            mask = np.logical_not(self.test_mask[other_node])

            # 条件匹配：test_mask 为 False 且 label 匹配
            same_label_mask = (labels == target_label) & mask

            # 按顺序提取
            same_label_nodes = other_node[same_label_mask]
            other_nodes = other_node[~same_label_mask]

            # 拼接结果
            other_node = np.concatenate([same_label_nodes, other_nodes])
            
            flag = False
            round +=1
            if round % 10000 == 0:
                
                print("round {} insertDE,check num: {}, reduce num: {} rate: {}".format(round,check_num,reduce_num,reduce_num/check_num))
                check_num =0
                reduce_num = 0
            for i in range(len(other_node)):
                
                add_edge_np = np.empty((200, 2), dtype=np.int32)
                current_index = 0

                dominate_flag = True # check if other_node[i] can connect to other nodes
                for j in range(len(other_node)):
                    if j == i:
                        continue
                    if other_node[j] not in self.list_of_set[other_node[i]]:
                        
                        is_dominated,dominating_edge_node = self.is_dominated_edge_return_node((other_node[i],other_node[j]))

                        # if len(NG_e1) != 1:
                        #     dominate_flag = False
                                
                        #     break
                        if not is_dominated:
                            
                            dominate_flag = False

                            break
                            
                            
                        else:
                            #add_edge.append((other_node[i],other_node[j]))
                            if dominating_edge_node == k:
                                add_edge_np[current_index] = [other_node[i], other_node[j]]
                                current_index += 1
                            
                            
                if dominate_flag == False:
                    continue
                else:
                    flag = True
                    dominating_node = other_node[i]
                
                    break
            if flag==False:
                # if k in  has_checked:
                #     print(f"has checked {k}")
                
                has_checked.add(k)
                continue
            else:
                reduce_num +=1
                if k in has_checked:
                    print(f"has checked {k}")
                add_edge = add_edge_np[:current_index]
                if self.node_degree[k] not in  degree:
                    degree[self.node_degree[k]] = 1
                else:
                    degree[self.node_degree[k]] += 1
                delete_node = True
                for e in add_edge:
                    # self.M[e[0], e[1]] = True
                    # self.M[e[1], e[0]] = True
                    self.list_of_set[e[0]].add(e[1])
                    self.list_of_set[e[1]].add(e[0])
                    self.remain_edges.add((e[0],e[1]))
                    
                    self.node_degree[e[0]] += 1
                    self.node_degree[e[1]] += 1
                    
                self.deleted_node[k] = True
                #self.need_delete.append(k)
                self.node_degree[k] = 1
                
                self.num_remain_nodes -= 1
        
                if self.label[k] == self.label[other_node[0] ]:
                    self.deleted_to_remain[k] = other_node[0] 
                    
                    self.label_list[other_node[0]]+=self.label_list[k]
                    
            
                else:
                    self.deleted_to_remain[k] = other_node[1]
                    
                    self.label_list[other_node[1]]+=self.label_list[k]
                    
                  
                #self.deleted_to_remain[k] = other_node[0]
                #print(self.num_reduced_node)
                
                if self.num_remain_nodes <= self.reduction_object:
                    self.finish_reduction = True
                    return 

                for neighbor in NG_row:
                    
                    self.node_degree[neighbor] -= 1
                    heapq.heappush(vertex_queue, (self.node_degree[neighbor], neighbor))    
                        
                heapq.heappush(vertex_queue, (self.node_degree[dominating_node], dominating_node))
                
                    
                if self.num_remain_nodes <= self.reduction_object:
                    self.finish = True
                    return
        
        print(degree)
        ed = time.time()
        self.heruistic_delete = True
        self.node_queue = deque(list(set(self.node_queue)))
        print("end insert dominated node,  node remain {} delete {} this round , time cost {}".format(self.num_remain_nodes,last_num_nodes-self.num_remain_nodes,ed-st)) 
        
        if delete_node == False:
            self.finish1 = True

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
            for j in self.M[i, :].nonzero()[1]:
                if i != j:
                    self.M[i, j] = False
                    self.M[j, i] = False
    

    def find_root(self, map, elem):
    # 找到最终指向自己的元素
        while map[elem] != elem:
            elem = map[elem]
        return elem


    def make_coarsened_graph_old(self):
        old_to_new = {}
        res = copy.deepcopy(self.data)

        for elem in self.deleted_to_remain.keys():
            root = self.find_root(self.deleted_to_remain, elem)
            self.deleted_to_remain[elem] = root
        
        edge_list = []
        new_node_num = sum(self.deleted_node == False)
        res['new_x'] = torch.zeros(new_node_num, self.data['x'].shape[1])
        index = 0

        # 映射未删除节点到新的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == False and self.keep_mask[i] == True:
                old_to_new[i] = index
                index += 1

        # 映射删除的节点到最终保留的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == True and self.keep_mask[i] == True:
                old_to_new[i] = old_to_new[self.deleted_to_remain[i]]

        # 创建新的节点到旧节点的映射
        new_to_old = {}
        for k, v in old_to_new.items():
            if v not in new_to_old.keys():
                new_to_old[v] = [k]
            else:
                new_to_old[v].append(k)
        
        # 添加边到新图中
        
        for i in range(len(self.list_of_set)):
            if self.deleted_node[i] == False:
                for j in self.list_of_set[i]:
                    if i != j and self.deleted_node[j] == False:
                        edge_list.append([old_to_new[i], old_to_new[j]])
        
        edge_list = np.array(edge_list).T

              
        
        # 计算新图的节点特征x为旧节点特征的平均
        for v in new_to_old.keys():
            res['new_x'][v] = torch.mean(self.data['x'][new_to_old[v]], dim=0)

        # 重置无关数据
        
        res['x'], res['y'], res['edge_index'], res['train_mask'], res['val_mask'], res['test_mask'] = 0, 0, 0, 0, 0, 0
        
        res['new_edge_index'] = torch.tensor(edge_list, dtype=torch.long)

        
        old_to_new = dict(sorted(old_to_new.items(), key=lambda item: item[1]))
        
        # adj_matrix_sparse =  build_sparse_adjacency_matrix(edge_list, len(new_to_old))
        # adj_matrix_sparse = adj_matrix_sparse.tolil()

        supernode_label = dict()
        label_list = dict()
        

        total_label_count = 0
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]
            total_label_count += len(label_list[key])
        print(f"total label {total_label_count}")
        #print(label_list)


        for key,value in label_list.items():
            if len(value) == 0:
                supernode_label[key] = 0
                continue
            values, counts = np.unique(value, return_counts=True)

            # 找出出现次数最多的元素
            max_label = values[np.argmax(counts)]
            supernode_label[key] = max_label

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1
        
        print(f"mis percentage {mis_label_count/total_label_count}")
        # for key,value in supernode_label.items():
        #     if value == -1:
        #         neighbor = adj_matrix_sparse.rows[key]
        #         neighbor_label = [supernode_label[n] for n in neighbor]
        #         if neighbor_label == []:
        #             supernode_label[key] = 0
        #         else:   
        #             values, counts = np.unique(neighbor_label, return_counts=True)
        #             max_label = supernode_label[neighbor[np.argmax(counts)]]
        #             supernode_label[key] = max_label
        
        res['node_label'] = supernode_label

        # homo = 0
        # for e in res.new_edge_index.T:
        #     if self.test_mask[int(e[0])] != -1  and self.test_mask[int(e[1])]!=-1:
        #         if supernode_label[int(e[0])] == supernode_label[int(e[1])]: 
        #             homo += 1
        # print(f"homo {homo/res.new_edge_index.shape[1]}")
        # 保存结果
        filename = f'./Reduced_Node_Data/{self.dataname}_{self.reduction_ratio:.2f}_split_All_Simplex.npy'
        np.save(filename, (res.cpu(), old_to_new))
        print(f"Saved to {filename}")
        f = open("./log.txt",'a')
        f.write(filename)
        f.close()
    
    def make_coarsened_graph_soft_label(self):
        old_to_new = {}
        res = copy.deepcopy(self.data)

        for elem in self.deleted_to_remain.keys():
            root = self.find_root(self.deleted_to_remain, elem)
            self.deleted_to_remain[elem] = root
        
        edge_list = []
        new_node_num = sum(self.deleted_node == False)
        res['new_x'] = torch.zeros(new_node_num, self.data['x'].shape[1])
        index = 0
        
        # 映射未删除节点到新的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == False and self.keep_mask[i] == True:
                old_to_new[i] = index
                index += 1

        # 映射删除的节点到最终保留的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == True and self.keep_mask[i] == True:
                old_to_new[i] = old_to_new[self.deleted_to_remain[i]]

        # 创建新的节点到旧节点的映射
        new_to_old = {}
        for k, v in old_to_new.items():
            if v not in new_to_old.keys():
                new_to_old[v] = [k]
            else:
                new_to_old[v].append(k)
        
        # 添加边到新图中
        
        for i in range(self.M.shape[0]):
            if self.deleted_node[i] == False:
                for j in self.M.rows[i]:
                    if i != j and self.deleted_node[j] == False:
                        edge_list.append([old_to_new[i], old_to_new[j]])
        
        edge_list = np.array(edge_list).T

            
        
        # 计算新图的节点特征x为旧节点特征的平均
        for v in new_to_old.keys():
            res['new_x'][v] = torch.mean(self.data['x'][new_to_old[v]], dim=0)

        # 重置无关数据
        
        res.x, res.y, res.edge_index, res.train_mask, res.val_mask, res.test_mask = 0, 0, 0, 0, 0, 0
        
        res.new_edge_index = torch.tensor(edge_list, dtype=torch.long)

        
        old_to_new = dict(sorted(old_to_new.items(), key=lambda item: item[1]))
        
        adj_matrix_sparse =  build_sparse_adjacency_matrix(edge_list, len(new_to_old))
        adj_matrix_sparse = adj_matrix_sparse.tolil()

        supernode_label = dict()
        label_list = dict()
        

        total_label_count = 0
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]
            
        print(f"total label {total_label_count}")
        #print(label_list)
        num_classes = np.max(self.label)+1
        one_hot_label = np.zeros((len(self.label), num_classes), dtype=np.int32)
        one_hot_label[np.arange(len(self.label)), self.label] = 1 
        
        for key,value in label_list.items():
            if len(value) == 0:
                supernode_label[key] = 0
                continue
            label = one_hot_label[label_list[key]]
            mix_label = np.sum(label,axis=0)
            # 找出出现次数最多的元素
            mix_label = mix_label / sum(mix_label)
            supernode_label[key] = mix_label

        # mis_label_count = 0
        # for key,value in label_list.items():
        #     for label in value:
        #         if label != supernode_label[key]:
        #             mis_label_count += 1
        
        # print(f"mis percentage {mis_label_count/total_label_count}")
        # for key,value in supernode_label.items():
        #     if value == -1:
        #         neighbor = adj_matrix_sparse.rows[key]
        #         neighbor_label = [supernode_label[n] for n in neighbor]
        #         if neighbor_label == []:
        #             supernode_label[key] = 0
        #         else:   
        #             values, counts = np.unique(neighbor_label, return_counts=True)
        #             max_label = supernode_label[neighbor[np.argmax(counts)]]
        #             supernode_label[key] = max_label
        
        res.node_label = supernode_label

        # homo = 0
        # for e in res.new_edge_index.T:
        #     if self.test_mask[int(e[0])] != -1  and self.test_mask[int(e[1])]!=-1:
        #         if supernode_label[int(e[0])] == supernode_label[int(e[1])]: 
        #             homo += 1
        # print(f"homo {homo/res.new_edge_index.shape[1]}")
        # 保存结果
        filename = f'/home/wuxiang/strong_collapse/Reduced_Node_Data/{self.dataname}_{self.reduction_ratio:.2f}_split_All_Simplex_1.npy'
        np.save(filename, (res.cpu(), old_to_new))
        print(f"Saved to {filename}")

    def make_coarsened_graph(self):
        old_to_new = {}
        res = copy.deepcopy(self.data)

        for elem in self.deleted_to_remain.keys():
            root = self.find_root(self.deleted_to_remain, elem)
            self.deleted_to_remain[elem] = root
        
        edge_list = []
        new_node_num = sum(self.deleted_node == False)
        res['new_x'] = torch.zeros(new_node_num, self.data['x'].shape[1])
        index = 0

        # 映射未删除节点到新的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == False and self.keep_mask[i] == True:
                old_to_new[i] = index
                index += 1

        # 映射删除的节点到最终保留的节点编号
        for i, v in enumerate(self.deleted_node):
            if v == True and self.keep_mask[i] == True:
                old_to_new[i] = old_to_new[self.deleted_to_remain[i]]

        # 创建新的节点到旧节点的映射
        new_to_old = {}
        for k, v in old_to_new.items():
            if v not in new_to_old.keys():
                new_to_old[v] = [k]
            else:
                new_to_old[v].append(k)
        
        # 添加边到新图中
        
        for i in range(self.M.shape[0]):
            if self.deleted_node[i] == False:
                for j in self.M.rows[i]:
                    if i != j and self.deleted_node[j] == False:
                        edge_list.append([old_to_new[i], old_to_new[j]])
        
        edge_list = np.array(edge_list).T

              
        
        # 计算新图的节点特征x为旧节点特征的平均
        

        
        old_to_new = dict(sorted(old_to_new.items(), key=lambda item: item[1]))
        
        adj_matrix_sparse =  build_sparse_adjacency_matrix(edge_list, len(new_to_old))
        adj_matrix_sparse = adj_matrix_sparse.tolil()
        

        old_to_new,new_to_old,label_list, supernode_label = self.swap_mis_label_any_node(old_to_new,new_to_old,adj_matrix_sparse)

        for v in new_to_old.keys():
            res['new_x'][v] = torch.mean(self.data['x'][new_to_old[v]], dim=0)

        # 重置无关数据
        
        res.x, res.y, res.edge_index, res.train_mask, res.val_mask, res.test_mask = 0, 0, 0, 0, 0, 0
        
        res.new_edge_index = torch.tensor(edge_list, dtype=torch.long)
        
        res.node_label = supernode_label

        # homo = 0
        # for e in res.new_edge_index.T:
        #     if self.test_mask[int(e[0])] != -1  and self.test_mask[int(e[1])]!=-1:
        #         if supernode_label[int(e[0])] == supernode_label[int(e[1])]: 
        #             homo += 1
        # print(f"homo {homo/res.new_edge_index.shape[1]}")
        # 保存结果
        filename = f'/home/wuxiang/strong_collapse/Reduced_Node_Data/{self.dataname}_{self.reduction_ratio:.2f}_split_All_Simplex_2.npy'
        # filename_res = f'/home/wuxiang/strong_collapse/Reduced_Node_Data/{self.dataname}_{self.reduction_ratio:.2f}_split_All_Simplex_1_res.npy'
        # filename_map = f'/home/wuxiang/strong_collapse/Reduced_Node_Data/{self.dataname}_{self.reduction_ratio:.2f}_split_All_Simplex_1_map.npy'
        # np.save(filename_res, res.cpu())
        # np.save(filename_map, old_to_new)
        np.save(filename,(res.cpu(),old_to_new))
        print(f"Saved to {filename}")
    
    def get_all_edges(self, inverse=False):
        upper_triangle = triu(self.M, k=1)
        rows, cols = upper_triangle.nonzero()
        if not inverse:
            return list(zip(rows, cols))
        else:
            # 包含双向边：每条边 (i, j) 和 (j, i)
            edges = list(zip(rows, cols))
            reverse_edges = list(zip(cols, rows))
            return edges + reverse_edges
    
    def get_all_edges_set(self):
        edge_set = set()
        for i in range(len(self.list_of_set)): 
            if self.deleted_node[i] == False:
                for j in self.list_of_set[i]:
                    if self.deleted_node[j]==False and (j,i) not in edge_set:       
                        edge_set.add((i,j))
        return edge_set
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
    
    def run_algorithm(self):
        
        while not self.finish:
            if self.heruistic_delete == False:
                self.node_queue = deque(self.keep_nodes)
            self.strong_collapse()
            
            if self.finish == True:
                break
            else:
                edges = self.get_all_edges()
                self.edge_queue = deque(edges) 
                self.edge_collapse()
                
                if self.finish == True:
                    break
            if self.insert_dominated_edge==True and self.heruistic_delete == False:
                self.insert_dominated_edges()
                print(f"insert dominated edge {self.num_reduced_node}")
                
            if self.heruistic_delete ==  True and self.finish == False:
                #self.heruistic_delete_edge()
                self.delete_deg2()
                print(f"break 4 ring {self.num_reduced_node}") 
            if self.heruistic_delete_deg3 ==  True  and self.finish == False:
                self.delete_deg3()
                print(f"delete degree 3 {self.num_reduced_node}")
        # for i in range(len(self.node_degree)):
        #     if self.node_degree[i] == 3 and self.deleted_node[i] == False:
        #         print("degree 3")
        if self.save == True:       
            self.make_coarsened_graph()

    def run_algorithm_relaxed_strong_collapse(self):
        is_first_strong_relaxed = True
        index = 0
        while  self.finish == False or  self.finish1 == False or self.finish2 == False:
            
            index += 1
            self.node_queue = deque([n for n in self.keep_nodes if not self.deleted_node[n]])
            self.strong_collapse()
            
            if self.finish_reduction == True:
                break
            else:
                #edges = self.get_all_edges()
                
                #self.edge_queue = deque(edges) 
                

                self.edge_collapse()
                self.isfirst = False
                
                
                
            
            
            if self.insert_dominated_edge==True and self.heruistic_delete==False:
                self.insert_dominated_edges_2()
                print(f"insert dominated edge remain {self.num_remain_nodes}")
                
                self.strong_relaxed = True
                self.exact_iter = index
                index = 0
                if self.finish_reduction == True:
                    break
                
                
            # if  self.strong_relaxed:
                
            #     if is_first_strong_relaxed:
            #         M1 = self.M.astype(int)
            #         self.adj_2_hop= M1 @ M1
                    
            #         self.adj_2_hop = self.adj_2_hop.tocsr()
            #         self.adj_2_hop.data -= 1
                    
            #         self.adj_2_hop.eliminate_zeros() 
            #         self.adj_2_hop = self.adj_2_hop.tolil()
                    
            #         self.adj_2_hop.astype(bool)
            #         self.strong_collapse_relaxed()
            #         self.heruistic_delete = True
            #         is_first_strong_relaxed = False
            #         if self.finish_reduction  == True:
            #             break

            # if self.heruistic_delete == True and len(self.potential_node_set) < 10000:
                
            #     self.delete_heterophlic_edge(10000)
        set_remain_edge = 0
        for id in range(len(self.list_of_set)):
            if self.deleted_node[id] == False:
                for node in self.list_of_set[id]:
                    if self.deleted_node[node] == False:
                        set_remain_edge += 1
        set_remain_edge = set_remain_edge / 2
        print(f"set remain edge {set_remain_edge}")      
        new_set = self.remain_edges.copy()
      
        for e in self.remain_edges:
            if self.deleted_node[e[0]] or self.deleted_node[e[1]]:
                new_set.remove(e) 
        self.remain_edges = new_set
        ori_remain_edge = self.remain_edges.copy()  
        ori_list_of_set = copy.deepcopy(self.list_of_set)
         
           
            #self.delete_heterophlic_edge(self.num_delete_hetero)
        
        self.remain_edges = ori_remain_edge.copy()
        self.list_of_set = copy.deepcopy(ori_list_of_set)
        self.delete_heterophlic_edge(self.num_delete_hetero )
        if self.save == True:    
            self.relax_iter = index   
            self.make_coarsened_graph_old()
    
    
    
    
    def run_algorithm_preserve_homotopy(self):
        
        while not self.finish and not self.finish1:
            
            #if self.heruistic_delete == False:
            st = time.time()
            self.node_queue = deque(self.keep_nodes)
            self.strong_collapse()
            
            if self.finish == True:
                break
            else:
                edges = self.get_all_edges()
                
                self.edge_queue = deque(edges) 
                self.edge_collapse()
                
                self.isfirst = False
                
                if self.finish == True:
                    break
            if self.insert_dominated_edge==True: # no potential node after intert dominated edge
                
                self.insert_dominated_edges_2()
                
                print(f"insert dominated edge {self.num_remain_nodes}")
                #self.draw_graph("collapse_1")
                break
        ed = time.time()
        print(f"all collapse time {ed-st}")
            
        # for i in range(len(self.node_degree)):
        #     if self.node_degree[i] == 3 and self.deleted_node[i] == False:
        #         print("degree 3")
        #for number in [20000,40000,60000,80000,100000,120000,140000,160000,180000,200000]:
        self.delete_heterophlic_edge(number)
        if self.save == True:       
            self.make_coarsened_graph()
        
        


#original_reuslt,old_map = np.load(f'./Reduced_Node_Data/{dataset_name}_0.30_split_All_Simplex_1.npy', allow_pickle=True)
if __name__ == "__main__":
    para_dict = {'Cora':{0.5:[15,15,400],0.3:[15,15,400],0.2:[15,15,600],0.1:[15,15,600]},
                 "Citeseer":{0.5:[15,15,1000],0.3:[15,15,2000],0.2:[15,15,700],0.1:[15,15,2000]},
                    'dblp':{0.5:[0,0],0.3:[10,10,0],0.2:[0,0],0.1:[25,25,2000]},
                 'ogbn-arxiv':{0.5:[25,25,40000],0.3:[15,15,2000],0.2:[15,15,200],0.1:[50,50,80000]},
                 'ogbn-products':{0.5:[100,100,0],0.3:[100,100,0],0.2:[100,100,0],0.1:[100,100,0]}}

    para_dict_APPNP = {'Cora':{0.5:[15,15,600],0.3:[15,15,2000],0.2:[15,15,200],0.1:[15,15,1000]},"Citeseer":{0.5:[15,15,900],0.3:[15,15,800],0.2:[15,15,700],0.1:[15,15,2000]}}
    degree_threshold =  {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Cora')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--deg1', type=int, default=100)
    parser.add_argument('--deg2', type=int, default=100)
    parser.add_argument('--del_edge', type=int, default=0)
    parser.add_argument('--compute_betti', type=bool, default=True)
    args = parser.parse_args()
    dataset_name = args.dataname
    args.deg1 = para_dict[args.dataname][args.ratio][0]
    args.deg2 = para_dict[args.dataname][args.ratio][1]
    args.del_edge = para_dict[args.dataname][args.ratio][2]
    print("dataname {}, ratio {}, deg1 {}, deg2 {}, del_edge {}".format(args.dataname,args.ratio,para_dict[args.dataname][args.ratio][0],para_dict[args.dataname][args.ratio][1],para_dict[args.dataname][args.ratio][2]))
    user = "wuxiang"
    if dataset_name == "Cora":
        data = torch.load('./dataset/Cora/processed/data.pt')[0]
        edges = data['edge_index']
        label = data.y
    elif dataset_name == "dblp":
        dataset = CitationFull(root='./dataset', name=dataset_name)
        data = dataset[0]
        
        edges = data['edge_index']
        label = data.y
    elif dataset_name == "Physics":
        dataset = Coauthor(root='/home/wuxiang/STPGC/dataset', name=dataset_name)
        data = dataset[0]
        label = data.y
        edges = data['edge_index']
    elif dataset_name == "Citeseer":
        dataset_path = './dataset/Citeseer'
        dataset =  Planetoid(root='./dataset/Citeseer', name='Citeseer')
        data = dataset[0]
        edges = data.edge_index
        label = data.y
    elif dataset_name == "pubmed":
        # dataset = Planetoid(root='./dataset/pubmed', name='Pubmed')
        # data = dataset[0]
        data = torch.load('/home/'+user+'/STPGC/dataset/pubmed/processed/data.pt')[0]
        edges = data['edge_index']
        label = data['y']
    elif dataset_name == "ogbn-arxiv":
        dataset_path = './dataset/arxiv'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/'+user+'/strong_collapse/dataset/arxiv')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-arxiv')
        data = dataset[0]
        edges = data['edge_index']
        reversed_edges = edges[[1, 0]]
        edges = torch.cat([edges, reversed_edges], dim=1)
    elif dataset_name == 'ogbn-products':
        dataset = PygNodePropPredDataset(name='ogbn-products', root='/mnt/ssd2/products/raw')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-products')
        data = dataset[0]
        edges = data['edge_index']
        reversed_edges = edges[[1, 0]]
    elif dataset_name == 'reddit':
        dataset = Reddit(root='/mnt/ssd2/Reddit/')

        data = dataset[0]
        edges = data['edge_index']
        label = data['y']
    
    # model = torch.load(f'./model_dir/mlp_model_{dataset_name}.pt')
    # model.eval()
    if dataset_name == 'dblp' or dataset_name == 'Physics':
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
    indices = []
    if dataset_name == "ogbn-arxiv" or dataset_name == "ogbn-products":
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
        data.y= data.y.view(-1)
        label = data.y
    
    num_classes = torch.unique(data['y'], return_counts=True)[0].shape[0]
    index_train = (data['train_mask'] == 1).nonzero().view(-1)
    for i in range(num_classes):
        index = (data['y'] == i).nonzero().view(-1)
        tensor_isin = torch.isin(index, index_train)
        index = index[tensor_isin]
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    keep_index = torch.cat([i[:int(i.size()[0] * 1.0)] for i in indices], dim=0)
    keep_mask = index_to_mask(keep_index, size=data['x'].size(0))
    node_mask = np.zeros(data['x'].size(0))

    if dataset_name == "ogbn-arxiv" or dataset_name == "reddit" :
        keep_nodes = np.arange(data.num_nodes)
    else:
        keep_nodes = find_component(data)
        print(len(keep_nodes))
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
 
    
    
    time_start = time.time()
    collapse = CoreAlgorithm(adj,data,keep_nodes,args.ratio,label,args,save = True)
    if args.ratio == 1.0:
        collapse.make_coarsened_graph_old()
    collapse.run_algorithm_relaxed_strong_collapse()
    time_end = time.time()
    print(f"total time {time_end - time_start}")
    f = open("./log.txt",'a')
    f.write(f"total time {time_end - time_start}\n")
    f.close()
    #collapse.draw_graph("collapsed_graph_figure_final.png")
    print("remain nodes :{}".format(collapse.num_remain_nodes))
    print("exact iter :{},relax iter {}".format(collapse.exact_iter,collapse.relax_iter))

    # if args.compute_betti:
    #     distance_matrix = np.inf * np.ones((num_nodes, num_nodes))
    #     for e in collapse.remain_edges:
            

    #         distance_matrix[e[0],e[1]] = 1
    #         distance_matrix[e[1],e[0]] = 1
    #     betti1,collapse_time_pers_time,diagram1,mem  = compute_diagram_ripser(distance_matrix)
    #     print(f"Betti 1 (环的数量): {betti1}")
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
