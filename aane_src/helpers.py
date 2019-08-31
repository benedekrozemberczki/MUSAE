import pandas as pd
import networkx as nx
import json
import numpy as np
from aane import AANE
from scipy import sparse
from scipy.sparse import csc_matrix
import argparse

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
                        default = './output/PTBR_aane.csv',
	                help = 'Output embedding.')

    return parser.parse_args()

def read_features(path):
    features = json.load(open(path))
    
    index_1 = [int(k) for k, v in features.items() for val in v]
    index_2 = [int(val) for k, v in features.items() for val in v]
    ones = [1.0]*len(index_1)
    node_count = max(index_1) + 1
    #if "croco" or "chameleon"  or "regr" in path:
    #    node_count=node_count + 1
    feature_count = max(index_2) + 1
    features = sparse.csc_matrix(sparse.coo_matrix((ones,(index_1,index_2)),shape=(node_count,feature_count),dtype=np.float32))
    return features

def read_graph(path):
    edges = pd.read_csv(path).values.tolist()
    edges = nx.from_edgelist(edges)
    A = sparse.csr_matrix(nx.adjacency_matrix(edges),dtype=np.float32)
    A.setdiag([0]*A.shape[0])
    A = csc_matrix(A)
    return A


def embedding_saver(H, path):
    columns = ["id"] + list(map(lambda x: "X_"+str(x),range(H.shape[1])))
    ids = np.array(range(0,H.shape[0])).reshape(-1,1)
    H = np.concatenate([ids, H],axis = 1)
    H = pd.DataFrame(H, columns = columns)
    H.to_csv(path, index = None) 
