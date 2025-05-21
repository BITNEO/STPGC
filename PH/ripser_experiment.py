



import numpy as np
from graph_coarsening_persistence_set import CoreAlgorithm
from scipy.sparse import eye, coo_matrix,triu,csr_matrix
import networkx as nx
import ripser

import json


import time

import pandas as pd
import csv
import torch

from compute_betti_number import compute_diagram_ripser
import os

import argparse
from memory_profiler import memory_usage
import tracemalloc
def compute_diagram(mat):
    distance_matrix = np.minimum(mat, mat.T)
    np.fill_diagonal(distance_matrix, 0)  # 对角线设置为0

    # 使用 ripser 计算持久图，包含 1 维同调类
    diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=1)['dgms']

    # 输出持久同调结果
    for i, diagram in enumerate(diagrams):
        
        print(f"H{i} 持久图:")
        print(diagram)
    return diagrams


    #st = time.time()
    simplex_tree.compute_persistence(persistence_dim_max=2)
    persistence_diagram = simplex_tree.persistence()
    #print(persistence_diagram)
    ed = time.time()
    #print(f"persistence:{ed-st}s")
    persistence_time = ed-st
    # print("extend PD")
    # st = time.time()
    # simplex_tree_pruned.extend_filtration()
    # dgms = simplex_tree_pruned.extended_persistence(min_persistence=1e-5)
    # # for (s,f) in simplex_tree_pruned.get_filtration():
    # #     print(s,f)
    # print(dgms)
    # ed = time.time()   
    print(f"extend persistence:{ed-st}s")
    return persistence_diagram,i,persistence_time

def compute_num_simplex_gudhi(distance_matrix,filatration_value_dict,deleted_node,max_filtration):
    # 设置足够大的 max_edge_length，以捕获环结构
    st = time.time()
    deleted_node = set(deleted_node)
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix,max_edge_length=max_filtration+100)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    ed = time.time()
    
    simplex_num = 0
    simplex_num_3 = 0
    simplex_num_2 = 0
    simplex_num_1 = 0
    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 3:
            simplex_num_3+=1
        elif len(simplex) == 2:
            simplex_num_2+=1
        elif len(simplex) == 1:
            simplex_num_1+=1
        simplex_num+=1
    print(f"simplex 3:{simplex_num_3},simplex 2:{simplex_num_2},simplex 1:{simplex_num_1}")
        
    print(simplex_num)
    #simplex_num = G.number_of_edges()+G.number_of_nodes()+sum(nx.triangles(G).values()) // 3
    return simplex_num


def compute_num_simplex(G):
    # 设置足够大的 max_edge_length，以捕获环结构
   
    simplex_num = G.number_of_edges()+G.number_of_nodes()+sum(nx.triangles(G).values()) // 3
    return simplex_num

# 估算环诞生的时刻，先删边再collpase掉其它的

class results:
    def __init__(self, index,original_time, collapse_time, collapse_persistence_time, original_num_node, original_num_edge,collapse_num_node,collapse_num_edge,original_mem,collapsed_mem,num_original_simplex,num_collapsed_simplex,algorithm_mem):
        # 初始化类的属性
        self.index = index
        self.original_time = original_time
        self.collapse_time = collapse_time
        self.collapse_persistence_time = collapse_persistence_time
        self.original_num_node = original_num_node
        self.original_num_edge = original_num_edge
        self.collapse_num_node = collapse_num_node
        self.collapse_num_edge = collapse_num_edge
        self.original_mem= original_mem
        self.collapsed_mem = collapsed_mem
        self.original_num_simplex = num_original_simplex
        self.collapse_num_simplex = num_collapsed_simplex
        self.algorithm_mem = algorithm_mem

       
        # 方法：返回一个简洁的描述
    def __str__(self):
        return (f"Graph Metrics:\n"
                f"Original Time: {self.original_time}\n"
                f"Collapse Time: {self.collapse_time}\n"
                f"Collapse Persistence Time: {self.collapse_persistence_time}\n"
                f"Original Number of Nodes: {self.original_num_node}\n"
                f"Original Number of Edges: {self.original_num_edge}")
        
def calculate_and_print_averages(results_list,dataname,k,k2):
    # 定义变量来累加各属性的值
    total_original_time = 0
    total_collapse_time = 0
    total_collapse_persistence_time = 0
    total_original_num_node = 0
    total_original_num_edge = 0
    total_collapse_num_node = 0
    total_collapse_num_edge = 0
    total_original_mem_use =0
    total_collapse_mem_use=0
    total_original_num_simplex = 0
    total_collapse_num_simplex = 0
    total_algorithm_mem = 0
    num_results = len(results_list)

    
    # 累加每个属性的值
    for result in results_list:
        total_original_time += result.original_time
        total_collapse_time += result.collapse_time
        total_collapse_persistence_time += result.collapse_persistence_time
        total_original_num_node += result.original_num_node
        total_original_num_edge += result.original_num_edge
        total_collapse_num_node += result.collapse_num_node
        total_collapse_num_edge += result.collapse_num_edge
        total_original_mem_use += result.original_mem
        total_collapse_mem_use += result.collapsed_mem
        total_original_num_simplex += result.original_num_simplex
        total_collapse_num_simplex += result.collapse_num_simplex
        total_algorithm_mem += result.algorithm_mem
            
        
        # 计算并打印每个属性的平均值
    print(dataname)
    print(f"Average Original Time: {total_original_time / num_results}")
    print(f"Average Collapse Time: {total_collapse_time / num_results}")
    print(f"Average Collapse Persistence Time: {total_collapse_persistence_time / num_results}")
    print(f"Average Collapse Total Time: {(total_collapse_time+total_collapse_persistence_time) / num_results}")
    print(f"Average Original Number of Nodes: {total_original_num_node / num_results}")
    print(f"Average Original Number of Edges: {total_original_num_edge / num_results}")
    print(f"Average Collapse Number of Nodes: {total_collapse_num_node / num_results}")
    print(f"Average Collapse Number of Edges: {total_collapse_num_edge / num_results}")
    print(f"Average Original Memory: {total_original_mem_use / num_results}")
    print(f"Average Collapse Memory: {total_collapse_mem_use / num_results}")
    print(f"Average Original Number of Simplex: {total_original_num_simplex / num_results}")
    print(f"Average Collapse Number of Simplex: {total_collapse_num_simplex / num_results}")
    print(f"Average Algorithm Memory: {total_algorithm_mem / num_results}")


    output_file = f"/home/xwu/strong_collapse_PH/PHresult/{dataname}{k}{k2}_ripser_persistence_result.csv"

# 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 写入文件
    with open(output_file, "w") as f:
        f.write(f"Average Original Time: {total_original_time / num_results}\n")
        f.write(f"Average Collapse Time: {total_collapse_time / num_results}\n")
        f.write(f"Average Collapse Persistence Time: {total_collapse_persistence_time / num_results}\n")
        f.write(f"Average Collapse Total Time: {(total_collapse_time+total_collapse_persistence_time) / num_results}")
        f.write(f"Average Original Number of Nodes: {total_original_num_node / num_results}\n")
        f.write(f"Average Original Number of Edges: {total_original_num_edge / num_results}\n")
        f.write(f"Average Collapse Number of Nodes: {total_collapse_num_node / num_results}\n")
        f.write(f"Average Collapse Number of Edges: {total_collapse_num_edge / num_results}\n")
        f.write(f"Average Original Memory: {total_original_mem_use / num_results}\n")
        f.write(f"Average Collapse Memory: {total_collapse_mem_use / num_results}\n")
        f.write(f"Average Original Number of Simplex: {total_original_num_simplex / num_results}\n")
        f.write(f"Average Collapse Number of Simplex: {total_collapse_num_simplex / num_results}\n")
        f.write(f"Average Algorithm Memory: {total_algorithm_mem / num_results}\n")

    print(f"结果已写入文件: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    dataname = "REDDIT5k"
    parser.add_argument("--dataset", type=str, required=False, help="dataset",default="REDDIT5k")
    parser.add_argument("--k", type=int, required=False, help="degree limit",default="6")
    parser.add_argument("--k2", type=int, required=False, help="degree limit",default="2")
    args = parser.parse_args()
    dataname = args.dataset
    

    if dataname == "REDDIT5k":
        if dataname == "REDDIT5k":
            edges =  pd.read_csv('./PH_dataset/REDDIT5K/REDDIT-MULTI-5K.edges', thousands=',')
            edges = edges.to_numpy()   
            graph_idx = pd.read_csv("./PH_dataset/REDDIT5K/REDDIT-MULTI-5K.graph_idx") 
        graph_idx = graph_idx.to_numpy()
        graphs = {}
        G = nx.Graph()
        G.add_edges_from(edges)
        G = G.to_undirected()
        components = list(nx.connected_components(G))
        graph = [com for com in components if len(com)>1000]
        subgraphs = []
        for index,com in enumerate(graph):
            subG = G.subgraph(com).copy()

            # 生成新 ID 映射，确保从 0 开始
            new_id_map = {old_id: new_id for new_id, old_id in enumerate(subG.nodes())}

            # 创建新的无向图，并添加重编号的边
            newG = nx.Graph()
            for u, v in subG.edges():
                newG.add_edge(new_id_map[u], new_id_map[v])

            subgraphs.append(newG)
            print(f"G {index}, nodes:{newG.number_of_nodes()},edges:{newG.number_of_edges()}")
            
        # for graph_id in np.unique(graph_idx):
        #     # 提取当前子图节点
        #     node_mask = (graph_idx == graph_id)
        #     subgraph_nodes = np.where(node_mask)[0]  # 节点原始ID
            
        #     # 筛选子图边
        #     edge_mask = np.isin(edges[:,0], subgraph_nodes) & np.isin(edges[:,1], subgraph_nodes)
        #     subgraph_edges = edges[edge_mask]
            
        #     # 创建NetworkX图对象
        #     G = nx.Graph()
        #     G.add_nodes_from(subgraph_nodes)
        #     G.add_edges_from(subgraph_edges)
            
        #     # 添加属性
        #     graphs[graph_id] = G
        #     print(f" nodes:{G.number_of_nodes()},edges:{G.number_of_edges()}")
    
            
    elif dataname == "malnet":
        graphs = []
        root_dir = "/home/xwu/strong_collapse_PH/PH_dataset/malnet-graphs-tiny/"
        for subdir, _, files in os.walk(root_dir):
            for subsubdir, _, subfiles in os.walk(subdir):
                for file in files:
                    if file.endswith(".edgelist"):
                        file_path = os.path.join(subdir, file)
                        
                        edges = []
                        unique_nodes = set()
                        with open(file_path, "r") as f:
                            for line in f:
                                if line.startswith("#"):
                                    continue  # 跳过注释行
                                u, v = map(int, line.strip().split())
                                edges.append((u, v))
                                unique_nodes.update([u, v])
                        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_nodes))}
                        graph = nx.Graph()
                        for u, v in edges:
                            graph.add_edge(node_mapping[u], node_mapping[v])
                        if graph.number_of_nodes()>1000:
                            graphs.append(graph)
                            print(f"Loaded graph  with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        subgraphs = graphs

    elif dataname == "oregon":
        graphs = []
        root_dir = "/home/xwu/strong_collapse_PH/PH_dataset/oregon/"
        for subdir, _, files in os.walk(root_dir):
            
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(subdir, file)
                        
                        edges = []
                        unique_nodes = set()
                        with open(file_path, "r") as f:
                            for line in f:
                                if line.startswith("#"):
                                    continue  # 跳过注释行
                                u, v = map(int, line.strip().split())
                                edges.append((u, v))
                                unique_nodes.update([u, v])
                        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_nodes))}
                        graph = nx.Graph()
                        for u, v in edges:
                            graph.add_edge(node_mapping[u], node_mapping[v])
                        if graph.number_of_nodes()>1000:
                            graphs.append(graph)
                            print(f"Loaded graph  with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                        

                
        subgraphs = graphs



    elif dataname == "enron":
        graphs = []
        root_dir = "/home/xwu/strong_collapse_PH/PH_dataset/enron/"
        for subdir, _, files in os.walk(root_dir):
            
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(subdir, file)
                        
                        edges = []
                        unique_nodes = set()
                        with open(file_path, "r") as f:
                            for line in f:
                                if line.startswith("#"):
                                    continue  # 跳过注释行
                                u, v = map(int, line.strip().split())
                                edges.append((u, v))
                                unique_nodes.update([u, v])
                        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_nodes))}
                        graph = nx.Graph()
                        for u, v in edges:
                            graph.add_edge(node_mapping[u], node_mapping[v])
                        if graph.number_of_nodes()>1000:
                            graphs.append(graph)
                            print(f"Loaded graph  with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                        

                
        subgraphs = graphs

    elif dataname == "p2p"  :
        graphs = []
        root_dir = f"/home/xwu/strong_collapse_PH/PH_dataset/{dataname}/"
        for subdir, _, files in os.walk(root_dir):
            
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(subdir, file)
                        
                        edges = []
                        unique_nodes = set()
                        with open(file_path, "r") as f:
                            for line in f:
                                if line.startswith("#"):
                                    continue  # 跳过注释行
                                u, v = map(int, line.strip().split())
                                edges.append((u, v))
                                unique_nodes.update([u, v])
                        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_nodes))}
                        graph = nx.Graph()
                        for u, v in edges:
                            graph.add_edge(node_mapping[u], node_mapping[v])
                        if graph.number_of_nodes()>1000:
                            graphs.append(graph)
                            print(f"Loaded graph  with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                        

                
        subgraphs = graphs

    elif dataname == "PDB_graph":
        graphs = []
        root_dir = f"/home/xwu/strong_collapse_PH/PH_dataset/{dataname}/"
        for subdir, _, files in os.walk(root_dir):
            
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(subdir, file)
                        
                        df = pd.read_csv(file_path)  # 确保文件路径正确
                        if df.shape[0] == 0:
                            continue
                        G = nx.Graph()  # 如果是有向图，改为 nx.DiGraph()
                        G.add_edges_from(df.values)
                        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
                        G_int = nx.relabel_nodes(G, node_mapping)
                        print(f"Loaded graph  with {G_int.number_of_nodes()} nodes and {G_int.number_of_edges()} edges.")
                        graphs.append(G_int)
                        if len(graphs) >= 100:
                            break

                        

                
        subgraphs = graphs

    # elif dataname == "FIRSTMM":
    #     f = open("/home/xwu/strong_collapse/PH_dataset/FIRSTMM-DB/FIRSTMM-DB.edges")
    #     f1 = open("/home/xwu/strong_collapse/PH_dataset/FIRSTMM-DB/FIRSTMM-DB.graph_idx")
    #     edges = f.readlines()
    #     graph_idx = f1.readlines()
    #     edges = [x.strip("\n") for x in edges]
    #     edges = [x.split(",") for x in edges]
    #     graph_idx = [x.strip("\n") for x in graph_idx]
    #     graph_idx = [int(x) for x in graph_idx]
    #     graph_index = 1
    #     subgraphs = []
    #     last_id = 0
    #     for i in range(len(edges)):
    #         edges[i] = [int(x) for x in edges[i]]
    #         if graph_idx[i] != graph_index:
    #             G = nx.Graph()
    #             G.add_edges_from(edges[last_id:i])
    #             subgraphs.append(G)
    #             last_id = i
    #             graph_index = graph_idx[i]

    # 添加节点和边
        

    elif dataname == "powerlaw":
        n =100
        # 生成一个cycle
        #G = nx.cycle_graph(n)
        subgraphs = []
        for i in range(1):
            G = nx.barabasi_albert_graph(n, 2)
            data = nx.node_link_data(G)
            with open("graph.json", "w") as f:
                json.dump(data, f)
            subgraphs.append(G)
    elif dataname == "test":

        with open("/home/xwu/strong_collapse/graph.json", "r") as f:
            data = json.load(f)
            G = nx.node_link_graph(data)
        subgraphs = [G]

    # 设置图像尺寸
    else:
        print("No dataset specified.")  
        exit(0)
    graph_list = subgraphs
    print(f"num graphs:{len(graph_list)}")
    # example_graph = graph_list[6]
    # data = nx.node_link_data(example_graph)
    # with open("graph1.json", "w") as f:
    #     json.dump(data, f)
    

    def wrapper(distance_matrix,filatration_value_dict):
        """调用要测试的函数"""
        return compute_diagram_ripser(distance_matrix,filatration_value_dict)

    # def wrapper(distance_matrix,filatration_value_dict):
    #     """调用要测试的函数"""
    #     return compute_diagram_ripser(distance_matrix,filatration_value_dict)

    def save_results_to_csv(results_list, filename):
        # 打开文件并准备写入
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入标题行
            writer.writerow([
                "index", "original_time", "collapse_time", "collapse_persistence_time",
                "original_num_node", "original_num_edge", "collapse_num_node", "collapse_num_edge"
                
            ])
            
            # 遍历每个Results对象并写入数据行
            for result in results_list:
                writer.writerow([
                    result.index,
                    result.original_time,
                    result.collapse_time,
                    result.collapse_persistence_time,
                    result.original_num_node,
                    result.original_num_edge,
                    result.collapse_num_node,
                    result.collapse_num_edge,
                    
                ])
    exp_result = []
    for out_index,G in enumerate(graph_list):
        # if out_index < 3223:
        #     continue
        n = G.number_of_nodes()
        edges = list(G.edges())
        
    # 找到最大节点 ID
        # if n>2000: # too large to compute
        #     continue
        data = nx.node_link_data(G)
        
        with open("/home/xwu/strong_collapse/graph.json", "w") as f:
            json.dump(data, f)
        
        
        distance_matrix = np.inf * np.ones((n, n))  # 初始化为无穷大

        # 第一个环 0-1-2-3-0
        filatration_value_dict = dict()
        tracemalloc.start()
        # if you want to compute extended persistence, you need to compute the birth time with the following and set value = max([n-e[0], n-e[1]]) to compute the death time
        print(f"original edges: {len(G.edges)}")
        G.remove_edges_from(nx.selfloop_edges(G))
        for i,e in enumerate(G.edges()):
            value = max([e[0], e[1]])
            G[e[0]][e[1]]['weight'] = value
            distance_matrix[e[0], e[1]] = value
            distance_matrix[e[1], e[0]] = value
            filatration_value_dict[e] = value
            filatration_value_dict[(e[1],e[0])] = value
        print("original diagram:")
    
        
        A = nx.to_scipy_sparse_array(G, format="lil")
        _,original_time,diagram0,mem  = compute_diagram_ripser(distance_matrix,filatration_value_dict)
        print(f"ripser time:{original_time}")
        num_original_simplex = compute_num_simplex(G)
        #num_original_simplex = compute_num_simplex_gudhi(distance_matrix,filatration_value_dict,[],value)
        original_mem =mem
        print(f"num original simplex:{num_original_simplex},num nodes {n}, num edges {len(G.edges)}")
        #mem_usage = memory_usage(lambda: wrapper(A,filatration_value_dict), interval=0.001)  # 每 0.1 秒记录一次内存
        current, original_mem = tracemalloc.get_traced_memory()
        original_mem = original_mem / (1024*1024)  # 转换为 MB
        tracemalloc.stop()
        print(f"max original memory: {original_mem} MB")
        
        #_,original_time,diagram0  = wrapper(distance_matrix,filatration_value_dict)
        #persistence_diagram_d,i,persistence_time = compute_diagram_dy(distance_matrix,filatration_value_dict)
        # for birth_death in persistence_diagram_d[1]:
        #     birth = birth_death.birth
        #     death = birth_death.death
        #     print(f"({birth:.3f}, {death:.3f})" if death != float('inf') else f"({birth:.3f}, ∞)")
        #print()
        #print(len(diagram2))
        #print(diagram0)
        
        #pos = nx.spring_layout(G)
        #plt.figure(figsize=(50, 50))
        #edge_labels = nx.get_edge_attributes(G, 'weight')
        #nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=300, font_size=8)

        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',font_size=8)
        #plt.savefig("collapsed_graph_figure_test"+".png", format="png")

        # subgraph_nodes = [1020,829,5249,305]
        # seed_nodes = 1020 
        # for node in G.neighbors(seed_nodes):
        #     subgraph_nodes.append(node)
        #     for j in G.neighbors(node):
        #         subgraph_nodes.append(j)
        #         for k in G.neighbors(j):
        #             subgraph_nodes.append(k)
        #             for l in G.neighbors(k):
        #                 subgraph_nodes.append(l)
        # subgraph_nodes = list(set(subgraph_nodes))        
        # node_color =  ['red' if n in [1020,829,5249,305] else  'lightblue' for n in subgraph_nodes]
        # # node_color[359] = 'red'
        # subgraph = G.subgraph(subgraph_nodes)
        # pos = nx.spring_layout(subgraph)
        # nx.draw(subgraph, pos,with_labels=True, node_color = node_color, edge_color="gray", node_size=500, font_size=10)
        # nx.draw_networkx_edge_labels(subgraph, pos, edge_labels= edge_labels,font_color='red',font_size=8)
        # plt.savefig("collapsed_graph_figure_test"+".png", format="png")

        tracemalloc.start()
        edges = np.array(G.edges())  # 直接转换为 NumPy 数组
        edges_reverse = edges[:, ::-1]  # 直接翻转列（交换 u, v）
        edges_with_reverse = np.vstack((edges, edges_reverse))  # 拼接正反向边

        data = np.ones(edges_with_reverse.shape[0], dtype=bool)  # 直接创建布尔数组
        rows, cols = edges_with_reverse.T  # 直接拆分 u, v
        # 创建 CSR 矩阵
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(len(G.nodes), len(G.nodes)))
        adj_matrix = adj_matrix + eye(n, format='coo')
        adj_matrix = adj_matrix.tolil()
        adj_matrix = adj_matrix.astype(bool)
        keep_node = np.arange(n)
        node_label = np.zeros(n, dtype=int)
        collapse = CoreAlgorithm(adj_matrix,0,keep_node,filatration_value_dict,0,keep_node,save=False,degree_threshold=args.k,degree_threshold2=args.k2)
        st = time.time()
        new_filtration_value_dict,new_edge,num_new_node,peak_mem = collapse.run_strong_edge_collapse()
        deleted_node = collapse.deleted_node
        coarsened_graph_mat = collapse.M
        ed = time.time()
        collapse_time=ed-st
        current, peak_mem, = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("peak memory usage:", peak_mem / (1024*1024), "MB")
        #deleted_edges = collapse.deleted_edges
        print("maximum memory in collapse:",peak_mem)
        #reduced_edges = collapse.get_all_edges()
        coarsened_graph = nx.Graph()
        coarsened_graph.add_edges_from(new_edge)
        # deleted_node = collapse.deleted_node
        # deleted_node_list = [i for i in range(n) if deleted_node[i] == True]
        #n = coarsened_graph.number_of_nodes()
        n = num_new_node

        tracemalloc.start()
        #collasped_adj = nx.to_scipy_sparse_matrix(coarsened_graph, format="lil")
        distance_matrix = np.inf * np.ones((n, n))
        for e in new_edge:
            

            distance_matrix[e[0],e[1]] = new_filtration_value_dict[(e[0],e[1])]
            distance_matrix[e[1],e[0]] = new_filtration_value_dict[(e[1],e[0])]
    
        #diagram1,num_simplices_collpased,collapse_time_pers_time = compute_diagram_gudhi(distance_matrix,new_filtration_value_dict,[],value)
        #A_collapsed = nx.to_scipy_sparse_matrix(coarsened_graph, format="lil")
        #mem_usage = memory_usage(lambda: wrapper(distance_matrix,filatration_value_dict), interval=0.1)  # 每 0.1 秒记录一次内存
        
        
        #print(len(diagram2))
        #collasped_memory = max(mem_usage) - min(mem_usage)
        _,collapse_time_pers_time,diagram1,mem  = compute_diagram_ripser(distance_matrix,new_filtration_value_dict)
        num_collapsed_simplex = compute_num_simplex(coarsened_graph)
        print(f"num collapsed simplex:{num_collapsed_simplex},num nodes {n}, num edges {len(coarsened_graph.edges)}")
        collasped_memory = mem
        
        G= coarsened_graph
        
        for i,e in enumerate(G.edges()):
            G[e[0]][e[1]]['weight'] = new_filtration_value_dict[(e[1],e[0])]

        # pos = nx.spring_layout(G)
        # plt.figure(figsize=(50, 50))
        # edge_labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw(G, pos,with_labels=True, node_color="lightblue", edge_color="gray", node_size=100, font_size=8)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',font_size=8)
        # plt.savefig("collapsed_graph_figure_collapsed"+".png", format="png")

        collapse.set_filtration_value_dict(filatration_value_dict)
        #approximate_PD = collapse.approximate_persistence()
        current, collasped_memory, = tracemalloc.get_traced_memory()
        collasped_memory = collasped_memory/(1024*1024)
        print(f"max collapsed memory: {collasped_memory} MB")
        tracemalloc.stop()

        # for e in reduced_edges:
        #     if filatration_value_dict[e] < 7:
        #         print(e)
        # plt.figure(figsize=(100, 100))
        # subgraph_nodes = [359,361]
        # seed_nodes = 359 
        # for node in G.neighbors(seed_nodes):
        #     subgraph_nodes.append(node)
        #     for j in G.neighbors(node):
        #         subgraph_nodes.append(j)
                # for k in G.neighbors(j):
                #     subgraph_nodes.append(k)
                    # for l in G.neighbors(k):
                    #     subgraph_nodes.append(l)
        # subgraph_nodes = list(set(subgraph_nodes))        
        # node_color =  ['red' if n == 359 or n==361 else  'lightblue' for n in subgraph_nodes]
        # # node_color[359] = 'red'
        # subgraph = G.subgraph(subgraph_nodes)
        # nx.draw(subgraph, pos,with_labels=True, node_color = node_color, edge_color="gray", node_size=500, font_size=10)
        # nx.draw_networkx_edge_labels(subgraph, pos, edge_labels= edge_labels,font_color='red',font_size=8)
        # plt.savefig("collapsed_graph_figure_collapsed"+".png", format="png")


        #print(len(diagram0) == len(diagram1) )
        diagram0 = diagram0[1]
        diagram1 = diagram1[1]
        # diagram0 = sorted(diagram0, key=lambda x: x[1][0]) 
        # diagram1 = sorted(diagram1, key=lambda x: x[1][0]) 
        diagram0 = np.array([a for a in diagram0 if a[1] == np.inf])
        diagram1 = np.array([a for a in diagram0 if a[1] == np.inf])
        #print(diagram0)
        # print(len(diagram0))
        # print(len(diagram1))
        if len(diagram0) != len(diagram1):
            print(f"different index {out_index}")
            break
        #print(diagram1)
        #print(approximate_PD)

        flag = True
        for j in range(len(diagram0)):
            

            if np.array_equal(diagram0[j][0], diagram1[j][0]) == False:
                print("False")
                print(f"{diagram0[j]},{diagram1[j]}")
                flag = False
        if flag ==True:
            print(f"{out_index} Two diagram are same")
        else:
            print("Two diagram are differnent")
            print(f"different index {out_index}")
            break
            f = open("./diagram_record.txt",'a')
            f.write(str(out_index)+"\n")
        flag = True
        result = results(index=out_index, original_num_node=len(deleted_node),original_num_edge=len(edges_with_reverse)/2,original_time=original_time,collapse_time=collapse_time,collapse_persistence_time=collapse_time_pers_time,\
                        collapse_num_node=n,collapse_num_edge=len(new_edge),original_mem=original_mem,collapsed_mem=collasped_memory,num_original_simplex = num_original_simplex,num_collapsed_simplex = num_collapsed_simplex,algorithm_mem = peak_mem) 
        exp_result.append(result)

    #save_results_to_csv(exp_result,"/home/xwu/strong_collapse_PH/PHresult/"+dataname+str(args.k)+"_ripser_persistence_result.csv")
    calculate_and_print_averages(exp_result,dataname,args.k,args.k2)
        
if __name__ == "__main__":
    main()