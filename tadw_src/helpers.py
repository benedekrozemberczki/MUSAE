import argparse
import networkx as nx
import pandas as pd
import numpy as np
import json
from scipy import sparse
from tqdm import tqdm
from texttable import Texttable
from sklearn.preprocessing import normalize
def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Wiki Chameleons.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run TADW.")


    parser.add_argument('--edge-path',
                        nargs = '?',
                        default = './input/edges/PTBR_edges.csv',
	                help = 'Input edges.')

    parser.add_argument('--feature-path',
                        nargs = '?',
                        default = './input/features/PTBR.json',
	                help = 'Input features.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './output/PTBR_tadw.csv',
	                help = 'Output embedding.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 64,
	                help = 'Number of dimensions. Default is 32.')

    parser.add_argument('--order',
                        type = int,
                        default = 2,
	                help = 'Target matrix approximation order. Default is 2.')

    parser.add_argument('--iterations',
                        type = int,
                        default = 50,
	                help = 'Number of gradient descent iterations. Default is 20.')

    parser.add_argument('--lambd',
                        type = float,
                        default = 0.2,
	                help = 'Regularization term coefficient. Default is 1000.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 10**-7,
	                help = 'Learning rate. Default is 10^-6.')

    parser.add_argument('--features',
                        nargs = '?',
                        default = 'sparse',
	                help = 'Output embedding.')

    parser.add_argument('--lower-control',
                        type = float,
                        default = 10**-15,
	                help = 'Overflow control. Default is 10**-15.')
    
    return parser.parse_args()

def normalize_adjacency(graph):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    A = sparse.coo_matrix(nx.adjacency_matrix(graph),dtype=np.float32)
    degs = sparse.coo_matrix((degs,(ind,ind)),shape=A.shape,dtype=np.float32)
    A = A.dot(degs)
    return A

def read_graph(edge_path, order):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param edge_path: Path to the ege list.
    :param order: Order of approximations.
    :return out_A: Target matrix.
    """
    print("Target matrix creation started.")
    graph = nx.from_edgelist(pd.read_csv(edge_path).values.tolist())
    A = normalize_adjacency(graph)
    A_sq = A.dot(A)
    out_A = (A+A_sq)/2

    return out_A

def read_sparse_features(feature_path):
    """
    Method to get sparse node feaures.
    :param feature_path:  Path to the node features.
    :return features: Node features.
    """
    features = json.load(open(feature_path))
    index_1 = [fet for k,v in features.items() for fet in v]
    index_2 = [int(k) for k,v in features.items() for fet in v]
    values = [1.0]*len(index_1) 
    nodes = list(map(lambda x: int(x), features.keys()))
    node_count = max(nodes)+1
    features = [list(map(lambda x: int(x),feature_set)) for node, feature_set in features.items()]
    feature_count = max(list(map(lambda x: max(x+[0]), features))) + 1
    features = sparse.coo_matrix((values,(index_1,index_2)),shape=(feature_count,node_count),dtype=np.float32)
    return features

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.items()])
    print(t.draw())
