import numpy as np
from scipy.sparse import lil_matrix
from ripser import ripser
import networkx as nx
import time
import tracemalloc
def compute_diagram_ripser(adj_matrix,filtration_value_dict):
    """
    计算给定稀疏邻接矩阵的 Betti 1 和 Betti 2 数。
    
    参数:
        adj_matrix (scipy.sparse.lil_matrix): 图的邻接矩阵，稀疏LIL格式。
        
    返回:
        tuple: 包含Betti 1和Betti 2数的元组。
    """
    # 将邻接矩阵转换为距离矩阵，非连通部分设置为无穷大
    peak=1
    tracemalloc.start()
    st = time.time()
    
    results = ripser(adj_matrix, distance_matrix=True,maxdim=1,thresh=1000000)
    dgms = results['dgms']

    # 计算 Betti 1 和 Betti 2
    betti_1 = len(dgms[1]) if len(dgms) > 1 else 0  # 环的数量
    #betti_2 = len(dgms[2]) if len(dgms) > 2 else 0  # 空腔的数量
    ed = time.time()
    print(f"persistence ripser:{ed-st}s")
    current, peak = tracemalloc.get_traced_memory()
    
    # peak = peak / 1024 ** 2
    # tracemalloc.stop() 
    return betti_1, ed-st,dgms,peak

# 测试函数
# 创建一个有5个节点的环的邻接矩阵
if __name__ == "__main__":
    G = nx.barabasi_albert_graph(10, 2)
    adj_matrix = nx.to_scipy_sparse_matrix(G, format='lil')
    betti_1, betti_2,diagrams = compute_betti_numbers(adj_matrix)
    print(f"Betti 1 (环的数量): {betti_1}")
    print(f"Betti 2 (空腔的数量): {betti_2}")
    for i, diagram in enumerate(diagrams):
        print(f"H{i} 持久图:")
        print(diagram)
